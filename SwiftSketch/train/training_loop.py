
import functools
import os
import blobfile as bf
import torch
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import re
from typing import Optional
from os.path import join as pjoin

from diffusion.resample import create_named_schedule_sampler
from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler


class TrainLoop:
    def __init__(self, args, model, diffusion, data, validation_data=None):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.validation_data = validation_data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.val_interval = args.val_interval or args.log_interval
        self.val_max_batches = args.val_max_batches
        self.val_seed = args.val_seed
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.init_checkpoint = args.init_checkpoint
        self.diffusion_mode = args.diffusion_mode
        self.cfm_instantaneous_prob = args.cfm_instantaneous_prob
        self.use_fp16 = False  
        self.fp16_scale_growth = 1e-3  
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr_schedule = args.lr_schedule
        self.lr_final_ratio = args.lr_final_ratio

        if self.lr_schedule == "exponential" and self.lr_anneal_steps:
            raise ValueError("--lr_schedule exponential cannot be combined with --lr_anneal_steps.")
        if self.lr_schedule == "exponential" and not 0 < self.lr_final_ratio < 1:
            raise ValueError("--lr_final_ratio must be greater than 0 and less than 1 for exponential scheduling.")

        self.step = 0
        self.resume_step = 0
        
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        # assert self.num_steps == 10_000
        # print(f"THEHEHEHEHEHE DATATATA SIZEZEZEZEZ ISISISISISISIS {len(self.data)}")
        # assert len(self.data) == 1_000
        # assert self.num_epochs == 11
   

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = None
        if self.diffusion_mode == "ddpm":
            self.schedule_sampler = create_named_schedule_sampler(
                self.schedule_sampler_type, diffusion
            )

        self.use_ddp = False
        

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.args.save_dir) or self.resume_checkpoint
        if resume_checkpoint and self.init_checkpoint:
            raise ValueError(
                "--init_checkpoint cannot be combined with a resume "
                "checkpoint or an existing checkpoint in the output directory."
            )
        if resume_checkpoint:
            self._validate_cfm_checkpoint_version(resume_checkpoint, "resume")
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print("resume_step" , self.resume_step, flush=True)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )
        elif self.init_checkpoint:
            self._validate_cfm_checkpoint_version(self.init_checkpoint, "initialize")
            logger.log(
                f"initializing model weights from checkpoint: {self.init_checkpoint}..."
            )
            state_dict = dist_util.load_state_dict(
                self.init_checkpoint, map_location=dist_util.dev()
            )
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            expected_missing = set()
            if self.diffusion_mode == "cfm_ddim":
                expected_missing = {
                    key for key in self.model.state_dict()
                    if key.startswith("embed_endpoint_timestep.")
                }
            if set(missing_keys) != expected_missing or unexpected_keys:
                raise ValueError(
                    "Checkpoint initialization had unexpected checkpoint keys. "
                    f"missing={missing_keys}, unexpected={unexpected_keys}"
                )
            if self.diffusion_mode == "cfm_ddim":
                self.model.initialize_endpoint_timestep()

    def _validate_cfm_checkpoint_version(self, checkpoint_path, action):
        """Reject CFM checkpoints from before the normalized-time format."""
        if self.diffusion_mode != "cfm_ddim":
            return
        checkpoint_args_path = bf.join(bf.dirname(checkpoint_path), "args.json")
        if not bf.exists(checkpoint_args_path):
            return
        with bf.BlobFile(checkpoint_args_path, "r") as checkpoint_args_file:
            import json
            checkpoint_args = json.load(checkpoint_args_file)
        if checkpoint_args.get("diffusion_mode") != "cfm_ddim":
            return
        if checkpoint_args.get("cfm_time_version") != 2:
            raise ValueError(
                f"Cannot {action} this CFM checkpoint: it predates the "
                "normalized-time implementation and is intentionally unsupported."
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.args.save_dir) or self.resume_checkpoint
        print("main_checkpoint", main_checkpoint, flush=True)
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)


    

    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}', flush=True)
            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                target_control_points, target_rendered_images, image_features = batch
                if self.diffusion_mode == "ddpm":
                    target_rendered_images = target_rendered_images.permute(0, 3, 1, 2).to(self.device)
                target_control_points = target_control_points.to(self.device)  # Move to device
                image_features=image_features.to(self.device) # Move to device

               
                self.run_step(target_rendered_images,target_control_points, image_features, step=self.step, resume_step=self.resume_step)
                
                  
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v), flush=True)

                        if k in ['step', 'samples']:
                            continue
                        else:
                            if self.args.use_wandb:
                                wandb.log({f'Loss/{k}': v}, step=self.step+self.resume_step)

                if self.validation_data is not None and self.step % self.val_interval == 0:
                    self.evaluate_validation(self.step + self.resume_step)


                if (self.step % self.save_interval == 0) and (self.step!=0):
                    self.save()
             
                self.step += 1

            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, target_rendered_images,target_control_points, image_features, step, resume_step):
        self.forward_backward( target_rendered_images,target_control_points, image_features, step, resume_step)
        if self.lr_schedule == "exponential":
            self._anneal_lr()
        self.mp_trainer.optimize(self.opt)
        if self.lr_schedule != "exponential":
            self._anneal_lr()
        self.log_step()

    def evaluate_validation(self, global_step):
        """Evaluate the full validation loss without changing model or RNG state."""
        was_training = self.model.training
        self.model.eval()
        totals = {}
        num_examples = 0
        num_batches = 0
        sample_index = 0

        try:
            with torch.no_grad():
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.val_seed)

                for batch in self.validation_data:
                    if self.val_max_batches and num_batches >= self.val_max_batches:
                        break

                    target_control_points, target_rendered_images, image_features = batch
                    if self.diffusion_mode == "ddpm":
                        target_rendered_images = target_rendered_images.permute(0, 3, 1, 2).to(self.device)
                    target_control_points = target_control_points.to(self.device)
                    image_features = image_features.to(self.device)

                    batch_size = target_control_points.shape[0]
                    noise = torch.randn(
                        target_control_points.shape,
                        dtype=target_control_points.dtype,
                        device=self.device,
                        generator=generator,
                    )

                    if self.diffusion_mode == "cfm_ddim":
                        time_t, time_r = sample_cfm_time_pairs(
                            batch_size,
                            1.0,
                            self.cfm_instantaneous_prob,
                            self.device,
                            generator=generator,
                        )
                        losses = self.diffusion.training_cfm_ddim_losses(
                            self.model,
                            target_control_points,
                            image_features,
                            time_t,
                            time_r,
                            noise=noise,
                        )
                    else:
                        timesteps = (
                            torch.arange(sample_index, sample_index + batch_size, device=self.device)
                            % self.diffusion.num_timesteps
                        ).long()
                        losses = self.diffusion.training_losses(
                            self.model,
                            target_control_points,
                            target_rendered_images,
                            image_features,
                            timesteps,
                            global_step,
                            self.resume_step,
                            noise=noise,
                            mode="eval",
                            log_results=False,
                        )

                    for key, value in losses.items():
                        totals[key] = totals.get(key, 0.0) + value.detach().float().mean().item() * batch_size
                    num_examples += batch_size
                    num_batches += 1
                    sample_index += batch_size
        finally:
            self.model.train(was_training)

        if num_examples == 0:
            print("Validation skipped: no batches were available.", flush=True)
            return

        validation_metrics = {
            f"Validation/{key}": total / num_examples for key, total in totals.items()
        }
        validation_metrics["Validation/num_examples"] = num_examples
        validation_metrics["Validation/num_batches"] = num_batches

        print(
            "step[{}]: validation_loss[{:0.5f}]".format(
                global_step, validation_metrics["Validation/loss"]
            ),
            flush=True,
        )
        if self.args.use_wandb:
            wandb.log(validation_metrics, step=global_step)

       
        

    def forward_backward(self, target_rendered_images,target_control_points, image_features, step, resume_step):
        self.mp_trainer.zero_grad()
        if self.diffusion_mode == "cfm_ddim":
            time_t, time_r = sample_cfm_time_pairs(
                target_control_points.shape[0],
                1.0,
                self.cfm_instantaneous_prob,
                dist_util.dev(),
            )
            losses = self.diffusion.training_cfm_ddim_losses(
                self.model,
                target_control_points,
                image_features,
                time_t,
                time_r,
            )
            log_cfm_time_pair_stats(time_t, time_r)
            log_loss_dict(self.diffusion, time_t, losses)
            self.mp_trainer.backward(losses["loss"].mean())
            return

        t, weights = self.schedule_sampler.sample(target_rendered_images.shape[0], dist_util.dev())
  
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            target_control_points, #  [batch_size, nstrokes, ncpoints, nfeats]
            target_rendered_images, #  [bs, canvas_height, canvas_width, 3]
            image_features, 
            t,  # [bs](int) sampled timesteps
            step,
            resume_step
          
        )
       

        losses = compute_losses()
        
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
   

        loss = (losses["loss"] * weights).mean()

        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if self.lr_schedule == "exponential":
            # Decay from the initial LR to --lr_final_ratio of it across this run.
            # The local step is intentional: each explicit continuation run gets
            # the schedule requested for its own --num_steps budget.
            progress = min(1.0, self.step / max(1, self.num_steps - 1))
            lr = self.lr * (self.lr_final_ratio ** progress)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
            return

        if not self.lr_anneal_steps:
            return
        # frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        frac_done = min(1.0, (self.step + self.resume_step) / self.lr_anneal_steps)
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(save_dir) -> Optional[str]:
    print("find_resume_checkpoint", flush=True)
    '''look for all file in save directory in the pattent of model{number}.pt
        and return the one with the highest step number.
    '''

    matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(save_dir)}
    models = {int(match.group(1)): file for file, match in matches.items() if match}

    return pjoin(save_dir, models[max(models)]) if models else None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())


def sample_cfm_time_pairs(
    batch_size,
    maximum_time,
    instantaneous_probability,
    device,
    generator=None,
):
    """Sample ordered continuous CFM times with clean data at time zero."""
    # Avoid the singular K coefficient exactly at the clean endpoint during
    # training. Inference is allowed to end at exactly zero.
    minimum_training_time = 1e-3
    random_times = torch.rand(
        batch_size, 2, device=device, generator=generator, dtype=torch.float32
    )
    random_times = minimum_training_time + random_times * (
        float(maximum_time) - minimum_training_time
    )
    time_t = random_times.max(dim=1).values
    time_r = random_times.min(dim=1).values
    instantaneous = torch.rand(
        batch_size, device=device, generator=generator
    ) < instantaneous_probability
    time_r = torch.where(instantaneous, time_t, time_r)
    return time_t, time_r


def log_cfm_time_pair_stats(time_t, time_r):
    distance = (time_t - time_r).detach()
    instantaneous = torch.isclose(time_t, time_r).float()
    logger.logkv_mean("cfm_time_t", time_t.detach().mean().item())
    logger.logkv_mean("cfm_time_r", time_r.detach().mean().item())
    logger.logkv_mean("cfm_instantaneous_fraction", instantaneous.mean().item())
    logger.logkv_mean("cfm_time_distance", distance.mean().item())
    logger.logkv_mean("cfm_time_distance_max", distance.max().item())
