# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""

import torch
import enum
import math
from contextlib import contextmanager

import numpy as np
import pydiffvg
import torch as th
from copy import deepcopy
from utils.sketch_utils import rander_image_from_points, render_image_from_norm_points, log_model_prediction, log_diffusion_process_to_wandb
from diffusion.Loss_computation import Loss, LPIPS


@contextmanager
def use_pydiffvg_device(device):
    """Temporarily select the renderer device and restore its global state."""
    previous_device = pydiffvg.get_device()
    pydiffvg.set_device(torch.device(device))
    try:
        yield
    finally:
        pydiffvg.set_device(previous_device)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1., cos_power=2):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** cos_power,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def sampling_snapshot_indices(total_steps, snapshot_count):
    """Return evenly spaced post-step indices, always including the final step.

    A sampling trajectory has ``total_steps`` post-step states.  This helper
    deliberately does not include the initial pure-noise state: requesting
    ``snapshot_count`` positions yields exactly that many trajectory states,
    with the final one included.
    """
    if snapshot_count < 0:
        raise ValueError("snapshot_count must be non-negative.")
    if snapshot_count == 0:
        return set()
    if snapshot_count > total_steps:
        raise ValueError("snapshot_count cannot exceed total sampling steps.")
    return {
        ((position + 1) * total_steps + snapshot_count - 1) // snapshot_count - 1
        for position in range(snapshot_count)
    }


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        args,
        *,
        betas,
        model_mean_type,
        model_var_type,
        rescale_timesteps=False,
        
    ):
        self.args=args
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps



        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
       
    
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # CFM uses a normalized continuous clock with x_0 equal to clean data.
        # The DDPM schedule table begins after its first noising increment, so
        # prepend alpha_bar(0)=1 for the clean endpoint.
        self.cfm_log_alphas_cumprod = th.from_numpy(
            np.log(np.concatenate(([1.0], self.alphas_cumprod)))
        ).float()
        self.cfm_betas = th.from_numpy(
            np.concatenate(([self.betas[0]], self.betas))
        ).float()
        self.cfm_numerical_eps = 1e-10


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.cfm_lpips_func = None
        if not hasattr(self.args, 'generate'):
            if getattr(self.args, "diffusion_mode", "ddpm") == "cfm_ddim":
                if getattr(self.args, "lpips_weight", 0.0) > 0:
                    self.cfm_lpips_func = LPIPS(args)
            else:
                self.loss_func = Loss(args)

       



    def plot_forward_pass_with_predicted_x0(self, model, data, device, scale):
        batch = next(iter(data))
        x_start, _, image_features = batch
        x_start= x_start.to(device)
        image_features=image_features.to(device)

        bs= x_start.shape[0]

        xt_to_plot = []
        predicted_x0_to_plot=[]


        freq= [0, 0.05,0.1,0.2,0.4,0.6,0.8, 1.0]
        timesteps = [math.floor(t * self.args.diffusion_steps) for t in freq]
      
        timesteps_for_log=[]

        for t in timesteps:
            if t==0:
                xt_sketch_list = render_image_from_norm_points(x_start, self.args.scaling_factor, self.args.canvas_width)
                xt_to_plot.append(xt_sketch_list)
                predicted_x0_to_plot.append(xt_sketch_list)
                timesteps_for_log.append(t)

            else:
                t = t-1 # x_t (in self.q_sample ) is calculated for t+1
                t_tensor = torch.tensor([t]*bs, dtype=torch.long).to(device)

                noise = torch.randn_like(x_start)  # shape [bs, nstrokes, ncpoints, nfeats]
                
                x_t = self.q_sample(x_start, t_tensor, noise=noise)  # shape [bs, nstrokes, ncpoints, nfeats]
                xt_sketch_list = render_image_from_norm_points(x_t, self.args.scaling_factor, self.args.canvas_width)
                
   
                xt_to_plot.append(xt_sketch_list)

                model_output = model(x=x_t, ts=self._scale_timesteps(t_tensor), image_features=image_features, scale=scale) #shape [bs,nstrokes, ncpoints, nfeats]
                
                if self.model_mean_type==ModelMeanType.EPSILON:
                    model_output_points= self._predict_xstart_from_eps(x_t=x_t, t=t_tensor, eps=model_output)
                else: #self.model_mean_type==ModelMeanType.START_X:
                    model_output_points= model_output

                predict_x0_sketch_list= render_image_from_norm_points(model_output_points, self.args.scaling_factor, self.args.canvas_width)
                
                predicted_x0_to_plot.append(predict_x0_sketch_list)
                t_for_log= self._scale_timesteps(t_tensor)[0]
                timesteps_for_log.append(t_for_log)
        
        if self.args.lpips_weight==0:
            predicted_x0_to_plot=[]

        log_diffusion_process_to_wandb(timesteps_for_log, xt_to_plot, predicted_x0_to_plot, "Forward Process Grid") 

      

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    # ------------------------------------------------------------------
    # Cumulative-flow DDIM helpers. These are separate from the legacy
    # discrete DDPM path so existing training and checkpoints remain intact.

    @staticmethod
    def _expand_batch_values(values, reference):
        """Expand one scalar per batch item to the rank of ``reference``."""
        return values.reshape(values.shape[0], *([1] * (reference.ndim - 1)))

    @staticmethod
    def _interpolate_schedule(values, times, maximum_time):
        """Piecewise-linearly interpolate a 1-D schedule at float times."""
        times = times.to(dtype=th.float32).clamp(0.0, float(maximum_time))
        values = values.to(device=times.device, dtype=times.dtype)
        lower = th.floor(times).long()
        upper = th.ceil(times).long()
        fraction = times - lower.to(times.dtype)
        return values[lower] + fraction * (values[upper] - values[lower])

    def cfm_alpha_bar_at(self, times):
        """Continuous alpha-bar with alpha_bar(0)=1 at the clean endpoint."""
        schedule_times = times * self.num_timesteps
        log_alpha_bar = self._interpolate_schedule(
            self.cfm_log_alphas_cumprod, schedule_times, self.num_timesteps
        )
        return th.exp(log_alpha_bar)

    def cfm_beta_at(self, times):
        """Continuous interpolation of beta in the clean-data-zero CFM clock."""
        schedule_times = times * self.num_timesteps
        return self._interpolate_schedule(
            self.cfm_betas, schedule_times, self.num_timesteps
        )

    def q_sample_cfm(self, x_0, times, noise=None):
        """Sample a noisy CFM state from clean control points x_0."""
        if noise is None:
            noise = th.randn_like(x_0)
        if noise.shape != x_0.shape:
            raise ValueError("CFM noise must have the same shape as x_0.")
        alpha_bar = self._expand_batch_values(
            self.cfm_alpha_bar_at(times), x_0
        )
        return th.sqrt(alpha_bar) * x_0 + th.sqrt(
            th.clamp(1.0 - alpha_bar, min=0.0)
        ) * noise

    def calculate_cfm_k(self, alpha_bar_t, alpha_bar_r, same_time=None):
        """Return the paper's scalar K(t,r), one value per batch item."""
        numerator = th.sqrt(th.clamp(1.0 - alpha_bar_t, min=0.0)) * th.sqrt(
            th.clamp(alpha_bar_r, min=self.cfm_numerical_eps)
        )
        denominator = th.sqrt(
            th.clamp(1.0 - alpha_bar_r, min=self.cfm_numerical_eps)
        ) * th.sqrt(th.clamp(alpha_bar_t, min=self.cfm_numerical_eps))
        coefficient_k = numerator / denominator - 1.0
        if same_time is not None:
            coefficient_k = th.where(
                same_time.to(dtype=th.bool),
                th.zeros_like(coefficient_k),
                coefficient_k,
            )
        return coefficient_k

    def calculate_cfm_state_tangent(self, x_0, x_t, alpha_bar_t):
        """Return v_x = sqrt(alpha_bar_t) x_0 - alpha_bar_t x_t."""
        alpha_bar_t = self._expand_batch_values(alpha_bar_t, x_0)
        return th.sqrt(alpha_bar_t) * x_0 - alpha_bar_t * x_t

    def calculate_cfm_time_coefficient(self, alpha_bar_t, beta_t):
        """Return the coefficient multiplying the active CFM time derivative.

        Public CFM time is normalized to s in [0, 1]. The DDIM coefficient is
        first expressed per internal schedule increment, so its JVP factor is
        divided by the number of schedule samples. This keeps the target
        stable when the same noise curve is discretized more finely.
        """
        coefficient = (
            2.0
            * (1.0 - alpha_bar_t)
            * (1.0 - beta_t)
            / th.clamp(beta_t, min=self.cfm_numerical_eps)
        )
        return coefficient / self.num_timesteps

    def predict_cfm_and_calculate_d(
        self,
        model,
        x_t,
        time_t,
        time_r,
        image_features,
        state_tangent,
        time_coefficient,
    ):
        """Return the network prediction and D using one joint JVP.

        D = J_x f v_x - c_t partial_t f. The endpoint tangent is zero,
        because r is held fixed in this directional derivative.
        """

        def model_at(state, current_time, endpoint_time):
            return model(
                x=state,
                timesteps=current_time,
                end_timesteps=endpoint_time,
                image_features=image_features,
            )

        # PyTorch 2.3's fused SDPA kernels do not implement forward AD.
        # Restrict only this JVP evaluation to the differentiable math kernel;
        # baseline DDPM and CFM inference retain their normal kernel choice.
        with th.nn.attention.sdpa_kernel(th.nn.attention.SDPBackend.MATH):
            prediction, derivative_d = th.func.jvp(
                model_at,
                (x_t, time_t, time_r),
                (
                    state_tangent,
                    -time_coefficient,
                    th.zeros_like(time_r),
                ),
            )
        return prediction, derivative_d

    @staticmethod
    def build_stopped_cfm_target(x_0, coefficient_k, derivative_d):
        """Build sg(x_0 + K D), with no gradient path through the target."""
        coefficient_k = GaussianDiffusion._expand_batch_values(
            coefficient_k, x_0
        )
        return (x_0 + coefficient_k * derivative_d).detach()

    def calculate_cfm_ddim_loss(
        self,
        prediction,
        x_0,
        alpha_bar_t,
        alpha_bar_r,
        derivative_d,
        time_t,
        time_r,
    ):
        """Pure tensor loss: no model is accepted or evaluated here."""
        coefficient_k = self.calculate_cfm_k(
            alpha_bar_t,
            alpha_bar_r,
            same_time=th.isclose(time_t, time_r),
        )
        stopped_target = self.build_stopped_cfm_target(
            x_0, coefficient_k, derivative_d
        )
        per_example_mse = (prediction - stopped_target).square().flatten(1).mean(1)
        return {
            "cfm_ddim_loss": per_example_mse,
            "loss": per_example_mse * self.args.cfm_loss_weight,
            "cfm_k_abs": coefficient_k.detach().abs(),
            "cfm_d_rms": derivative_d.detach().square().flatten(1).mean(1).sqrt(),
        }

    @staticmethod
    def select_cfm_lpips_prediction(
        model,
        cfm_prediction,
        x_t,
        time_t,
        time_r,
        image_features,
    ):
        """Use t=r predictions directly and predict r=0 for cumulative rows."""
        cumulative = ~th.isclose(time_t, time_r)
        if not th.any(cumulative):
            return cfm_prediction

        terminal_prediction = model(
            x=x_t[cumulative],
            timesteps=time_t[cumulative],
            end_timesteps=th.zeros_like(time_r[cumulative]),
            image_features=(
                None if image_features is None else image_features[cumulative]
            ),
        )
        lpips_prediction = cfm_prediction.clone()
        lpips_prediction[cumulative] = terminal_prediction
        return lpips_prediction

    def calculate_cfm_lpips_loss(
        self,
        prediction,
        target_rendered_images,
        mode,
        render_device,
    ):
        """Render a precomputed prediction and calculate raw SwiftSketch LPIPS."""
        if self.cfm_lpips_func is None:
            raise ValueError("CFM LPIPS was requested without an LPIPS loss module.")
        if target_rendered_images is None:
            raise ValueError("CFM LPIPS requires rendered ground-truth sketches.")

        canvas_points = prediction / self.args.scaling_factor
        canvas_points = (canvas_points + 1.0) / 2.0
        canvas_points = canvas_points * self.args.canvas_width
        with use_pydiffvg_device(render_device):
            rendered_prediction, _ = rander_image_from_points(
                canvas_points,
                self.args.canvas_width,
                self.args.canvas_height,
            )
        rendered_prediction = rendered_prediction.permute(0, 3, 1, 2)
        target_rendered_images = target_rendered_images.to(
            rendered_prediction.device
        ).detach()
        return self.cfm_lpips_func(
            rendered_prediction,
            target_rendered_images,
            mode=mode,
        ).mean()

    def training_cfm_ddim_losses(
        self,
        model,
        x_0,
        image_features,
        time_t,
        time_r,
        noise=None,
        target_rendered_images=None,
        mode="train",
        render_device=None,
    ):
        """Calculate the CFM prediction/JVP first, then call the tensor loss."""
        if self.model_mean_type != ModelMeanType.START_X:
            raise ValueError("CFM-DDIM currently supports only x_0 prediction.")

        time_t = time_t.to(device=x_0.device, dtype=th.float32)
        time_r = time_r.to(device=x_0.device, dtype=th.float32)
        if th.any(time_r > time_t):
            raise ValueError("Every CFM endpoint r must satisfy r <= t.")
        if noise is None:
            noise = th.randn_like(x_0)

        x_t = self.q_sample_cfm(x_0, time_t, noise=noise)
        alpha_bar_t = self.cfm_alpha_bar_at(time_t)
        alpha_bar_r = self.cfm_alpha_bar_at(time_r)
        beta_t = self.cfm_beta_at(time_t)
        state_tangent = self.calculate_cfm_state_tangent(
            x_0, x_t, alpha_bar_t
        )
        time_coefficient = self.calculate_cfm_time_coefficient(
            alpha_bar_t, beta_t
        )

        prediction, derivative_d = self.predict_cfm_and_calculate_d(
            model,
            x_t,
            time_t,
            time_r,
            image_features,
            state_tangent,
            time_coefficient,
        )
        # D is a self-estimated target component. Stop its graph before it
        # crosses the prediction/JVP boundary into the pure loss function;
        # build_stopped_cfm_target also detaches the complete target as a
        # defensive guarantee.
        stopped_derivative_d = derivative_d.detach()
        losses = self.calculate_cfm_ddim_loss(
            prediction,
            x_0,
            alpha_bar_t,
            alpha_bar_r,
            stopped_derivative_d,
            time_t,
            time_r,
        )
        losses["cfm_time_coefficient_abs"] = time_coefficient.detach().abs()
        if self.cfm_lpips_func is not None:
            lpips_prediction = self.select_cfm_lpips_prediction(
                model,
                prediction,
                x_t,
                time_t,
                time_r,
                image_features,
            )
            lpips_loss = self.calculate_cfm_lpips_loss(
                lpips_prediction,
                target_rendered_images,
                mode,
                x_0.device if render_device is None else render_device,
            )
            losses["lpips_loss"] = lpips_loss
            losses["loss"] = (
                losses["loss"] + self.args.lpips_weight * lpips_loss
            )
        return losses

    def ddim_cumulative_step(self, x_t, x_hat_t_to_r, time_t, time_r):
        """Apply the deterministic cumulative DDIM map F(x_hat, x_t, t, r)."""
        alpha_bar_t = self._expand_batch_values(
            self.cfm_alpha_bar_at(time_t), x_t
        )
        alpha_bar_r = self._expand_batch_values(
            self.cfm_alpha_bar_at(time_r), x_t
        )
        predicted_noise = (
            x_t - th.sqrt(alpha_bar_t) * x_hat_t_to_r
        ) / th.sqrt(
            th.clamp(1.0 - alpha_bar_t, min=self.cfm_numerical_eps)
        )
        mapped = (
            th.sqrt(alpha_bar_r) * x_hat_t_to_r
            + th.sqrt(th.clamp(1.0 - alpha_bar_r, min=0.0)) * predicted_noise
        )
        same_time = self._expand_batch_values(
            th.isclose(time_t, time_r), x_t
        )
        return th.where(same_time, x_t, mapped)

    def cfm_ddim_sample_loop(
        self,
        model,
        shape,
        image_features,
        num_steps,
        noise=None,
        scale=None,
        device=None,
        progress=False,
        return_intermediates=False,
        intermediate_steps=0,
    ):
        """Generate with exactly ``num_steps`` deterministic CFM-DDIM hops.

        When requested, return saved post-hop states and their corresponding
        clean predictions alongside the final sample.  Saved tensors live on
        CPU so they do not retain GPU memory during long trajectories.
        """
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError("CFM sampling steps must be in [1, diffusion_steps].")
        snapshot_indices = sampling_snapshot_indices(num_steps, intermediate_steps)
        if device is None:
            device = next(iter(model.parameters())).device
        if noise is None:
            current = th.randn(*shape, device=device)
        else:
            if tuple(noise.shape) != tuple(shape):
                raise ValueError("Initial CFM noise shape does not match requested shape.")
            current = noise.to(device)

        time_grid = th.linspace(
            1.0,
            0.0,
            num_steps + 1,
            device=device,
            dtype=th.float32,
        )
        hop_indices = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            hop_indices = tqdm(hop_indices)

        intermediates = []
        with th.no_grad():
            for hop in hop_indices:
                time_t = time_grid[hop].expand(shape[0])
                time_r = time_grid[hop + 1].expand(shape[0])
                prediction = model(
                    x=current,
                    timesteps=time_t,
                    end_timesteps=time_r,
                    image_features=image_features,
                    scale=scale,
                )
                current = self.ddim_cumulative_step(
                    current, prediction, time_t, time_r
                )
                if hop in snapshot_indices:
                    intermediates.append({
                        "state": current.detach().cpu().clone(),
                        "prediction": prediction.detach().cpu().clone(),
                        "step": hop + 1,
                        "total_steps": num_steps,
                    })
        if return_intermediates:
            return current, intermediates
        return current

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, image_features= None,
            scale= None 
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param image_features: features used to condition the model's prediction. 
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
      
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x=x, ts=self._scale_timesteps(t),image_features=image_features, scale=scale)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]



            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output  
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else: # self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            
                    


            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

          

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        image_features= None,
        scale= None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param image_features: features used to condition the model's prediction.    
        :param scale: guidance_param
        :param const_noise: If True, will noise all samples with the same noise throughout sampling


        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            image_features= image_features,
            scale= scale,
 
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        image_features= None,
        scale= None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        dump_steps=None,
        const_noise=False,
        return_intermediates=False,
        intermediate_steps=0,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).  #(args.batch_size, args.num_paths, model.npoints, model.nfeats)
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param image_features: features used to condition the model's prediction.    
        :param scale: guidance_param
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        total_steps = self.num_timesteps - skip_timesteps
        snapshot_indices = sampling_snapshot_indices(total_steps, intermediate_steps)
        if return_intermediates and dump_steps is not None:
            raise ValueError("dump_steps and return_intermediates cannot be used together.")
        final = None
        intermediates = []
        if dump_steps is not None:
            dump = []


        freq= [ 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0]

        timesteps = [math.floor(t * self.args.diffusion_steps) for t in freq]
    
        if self.args.use_wandb:
            timesteps_to_save = [timesteps[0] - t for t in timesteps[:-1]]
            timesteps_to_save.reverse()
 
        xt_Denoising_Process=[] #list of lists
        x0_Denoising_Process=[] #list of lists

 
        for i, out in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            image_features= image_features,
            scale= scale,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            const_noise=const_noise,
        )):
                   
            if self.args.use_wandb:
                if i in timesteps_to_save:
                    xt_sketch_list = render_image_from_norm_points(out["image"], self.args.scaling_factor,  self.args.canvas_width)
                    xt_Denoising_Process.append(xt_sketch_list)
                    x0_sketch_list = render_image_from_norm_points(out["pred_xstart"], self.args.scaling_factor, self.args.canvas_width)
                    x0_Denoising_Process.append(x0_sketch_list)

                if i== self.args.diffusion_steps-1:
                    xt_sketch_list = render_image_from_norm_points(out["sample"], self.args.scaling_factor, self.args.canvas_width)
                    xt_Denoising_Process.append(xt_sketch_list)
                    x0_Denoising_Process.append(xt_sketch_list)


            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(out["sample"]))
            if i in snapshot_indices:
                intermediates.append({
                    "state": out["sample"].detach().cpu().clone(),
                    "prediction": out["pred_xstart"].detach().cpu().clone(),
                    "step": i + 1,
                    "total_steps": total_steps,
                })
            final = out
        if dump_steps is not None:
            return dump
 
        
        if self.args.use_wandb:
            if self.args.lpips_weight==0:
                x0_Denoising_Process=[]


            print("log Denoising Process")
            log_diffusion_process_to_wandb(timesteps[::-1], xt_Denoising_Process[::-1], x0_Denoising_Process[::-1], "Denoising Process Grid") 

        if return_intermediates:
            return final["sample"], intermediates
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        image_features= None,
        scale= None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(iter(model.parameters())).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
 
            with th.no_grad():

                sample_fn =  self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    image_features= image_features,
                    scale= scale,
                    const_noise=const_noise,
                )
                out["image"]= img

                yield out
                img = out["sample"]
                


    def training_losses(
        self,
        model,
        x_start,
        x_start_randered_images,
        image_features,
        t,
        step,
        resume_step,
        noise=None,
        mode="train",
        log_results=True,
    ):

        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. (control points)
        :param x_start_randered_image: the [N x C x ...] tensor of inputs_randered_images. (the sketches for image loss)
        :param t: a batch of timestep indices.
        :param image_features : a batch of image features (the condition)
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

    
        if noise is None:
            noise = th.randn_like(x_start)  #shape [bs,nstrokes, ncpoints, nfeats]
        x_t = self.q_sample(x_start, t, noise=noise) #shape [bs,nstrokes, ncpoints, nfeats]
  
        terms = {}
      
        model_output = model(x=x_t, ts=self._scale_timesteps(t), image_features=image_features) #shape [bs,nstrokes, ncpoints, nfeats]
       
        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise

        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs,nstrokes, ncpoints, nfeats]

        #get predicted points from model output

        if self.model_mean_type==ModelMeanType.START_X or self.args.lpips_weight or self.args.clip_conv_weight:
        
            if self.model_mean_type==ModelMeanType.EPSILON:
                model_output_points= self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
            else: #self.model_mean_type==ModelMeanType.START_X:
                model_output_points= model_output


            if log_results and self.args.use_wandb: #log model prediction

                if step % self.args.log_interval == 0:
                    x0_sketch= render_image_from_norm_points(x_start[0].unsqueeze(0), self.args.scaling_factor, self.args.canvas_width)[0]
                    xt_sketch= render_image_from_norm_points(x_t[0].unsqueeze(0), self.args.scaling_factor, self.args.canvas_width)[0]
                    predict_x0_sketch= render_image_from_norm_points(model_output_points[0].unsqueeze(0), self.args.scaling_factor, self.args.canvas_width)[0]
                    t_for_log= self._scale_timesteps(t)[0]
                    quartile = int(4 * t_for_log / self.num_timesteps)

                    log_model_prediction(x0_sketch, xt_sketch, predict_x0_sketch, t_for_log, quartile, step+resume_step)


        if self.args.lpips_weight or self.args.clip_conv_weight:
            #convert the normalized points back to the original range for rendering 
            unnormalized_model_output_points = model_output_points / self.args.scaling_factor
            unnormalized_model_output_points = (unnormalized_model_output_points + 1) / 2
            unnormalized_model_output_points = unnormalized_model_output_points * self.args.canvas_width

            output_rendered_images, _= rander_image_from_points(unnormalized_model_output_points,self.args.canvas_width, self.args.canvas_height)
            output_rendered_images = output_rendered_images.permute(0 ,3, 1, 2).to(x_start.device)
        else:
            output_rendered_images= torch.tensor([0.0]).to(x_start.device)

        #reshape the points for loss
        bs, nstrokes, ncpoints, nfeats = model_output.shape
        model_output= model_output.reshape(bs, nstrokes, ncpoints*nfeats)# [bs,nstrokes, ncpoints*nfeats (8)]
        target= target.reshape(bs, nstrokes, ncpoints*nfeats)# [bs,nstrokes, ncpoints*nfeats (8)]

        terms = self.loss_func(
            output_rendered_images,
            x_start_randered_images.detach(),
            model_output,
            target.detach(),
            mode=mode,
        )
        terms["loss"] = sum(list(terms.values())) 
        return terms
    



def _extract_into_tensor(arr, timesteps, broadcast_shape):

    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
