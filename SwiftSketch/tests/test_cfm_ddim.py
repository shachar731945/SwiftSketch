import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from model.SwiftSketch_model import SwiftSketch
from model.cfg_sampler import ClassifierFreeSampleModel


def make_diffusion(num_steps=8):
    args = SimpleNamespace(
        diffusion_mode="cfm_ddim",
        cfm_loss_weight=1.0,
    )
    return GaussianDiffusion(
        args=args,
        betas=np.linspace(1e-3, 2e-2, num_steps),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
    )


def make_normalized_cosine_diffusion(num_steps):
    args = SimpleNamespace(
        diffusion_mode="cfm_ddim",
        cfm_loss_weight=1.0,
    )
    return GaussianDiffusion(
        args=args,
        betas=get_named_beta_schedule("cosine", num_steps, cos_power=0.4),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
    )


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = 0

    def forward(
        self,
        x,
        timesteps,
        image_features=None,
        end_timesteps=None,
        scale=None,
    ):
        self.calls += 1
        time = timesteps.reshape(-1, *([1] * (x.ndim - 1)))
        endpoint = end_timesteps.reshape(-1, *([1] * (x.ndim - 1)))
        return self.weight * x + 0.01 * (time + endpoint)


class RecordingConditionalModel(CountingModel):
    def __init__(self):
        super().__init__()
        self.cond_mask_prob = 0.1
        self.cond_mode = "image"
        self.ncpoints = 4
        self.nfeats = 2
        self.endpoint_records = []

    def forward(
        self,
        x,
        timesteps,
        image_features=None,
        end_timesteps=None,
        uncond=False,
        scale=None,
    ):
        self.endpoint_records.append((end_timesteps.clone(), uncond))
        output = super().forward(
            x,
            timesteps,
            image_features=image_features,
            end_timesteps=end_timesteps,
            scale=scale,
        )
        return output - 0.1 if uncond else output


class CFMDDIMTests(unittest.TestCase):
    def test_clean_data_is_time_zero(self):
        diffusion = make_diffusion()
        x_0 = torch.randn(2, 3, 4, 2)
        noise = torch.randn_like(x_0)
        time_zero = torch.zeros(2)
        self.assertTrue(
            torch.equal(diffusion.q_sample_cfm(x_0, time_zero, noise), x_0)
        )
        self.assertTrue(
            torch.equal(
                diffusion.cfm_alpha_bar_at(time_zero), torch.ones(2)
            )
        )

    def test_normalized_cfm_schedule_is_stable_when_resolution_changes(self):
        diffusion_50 = make_normalized_cosine_diffusion(50)
        diffusion_200 = make_normalized_cosine_diffusion(200)
        time_t = torch.tensor([0.2, 0.7])
        time_r = torch.tensor([0.1, 0.3])

        alpha_t_50 = diffusion_50.cfm_alpha_bar_at(time_t)
        alpha_t_200 = diffusion_200.cfm_alpha_bar_at(time_t)
        alpha_r_50 = diffusion_50.cfm_alpha_bar_at(time_r)
        alpha_r_200 = diffusion_200.cfm_alpha_bar_at(time_r)
        self.assertTrue(torch.allclose(alpha_t_50, alpha_t_200, atol=1e-6))
        self.assertTrue(torch.allclose(alpha_r_50, alpha_r_200, atol=1e-6))

        k_50 = diffusion_50.calculate_cfm_k(alpha_t_50, alpha_r_50)
        k_200 = diffusion_200.calculate_cfm_k(alpha_t_200, alpha_r_200)
        self.assertTrue(torch.allclose(k_50, k_200, atol=1e-6))

        coefficient_50 = diffusion_50.calculate_cfm_time_coefficient(
            alpha_t_50, diffusion_50.cfm_beta_at(time_t)
        )
        coefficient_200 = diffusion_200.calculate_cfm_time_coefficient(
            alpha_t_200, diffusion_200.cfm_beta_at(time_t)
        )
        # beta is still the finite DDIM schedule increment, so the two
        # resolutions agree up to its expected discretization error.
        self.assertTrue(torch.allclose(coefficient_50, coefficient_200, rtol=0.05))

    def test_ddim_map_is_identity_when_endpoint_equals_current_time(self):
        diffusion = make_diffusion()
        x_t = torch.randn(2, 3, 4, 2)
        prediction = torch.randn_like(x_t)
        times = torch.tensor([0.0, 0.525])
        mapped = diffusion.ddim_cumulative_step(
            x_t, prediction, times, times
        )
        self.assertTrue(torch.equal(mapped, x_t))

    def test_instantaneous_loss_is_x0_mse_and_target_is_stopped(self):
        diffusion = make_diffusion()
        prediction = torch.randn(2, 3, 4, 2, requires_grad=True)
        derivative_d = torch.randn_like(prediction, requires_grad=True)
        x_0 = torch.randn_like(prediction)
        times = torch.tensor([0.1, 0.4])
        alpha_bar = diffusion.cfm_alpha_bar_at(times)

        terms = diffusion.calculate_cfm_ddim_loss(
            prediction,
            x_0,
            alpha_bar,
            alpha_bar,
            derivative_d,
            times,
            times,
        )
        expected = (prediction - x_0).square().flatten(1).mean(1)
        self.assertTrue(torch.allclose(terms["cfm_ddim"], expected))
        self.assertTrue(torch.equal(terms["cfm_k_abs"], torch.zeros(2)))

        terms["loss"].mean().backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(derivative_d.grad)

    def test_endpoint_embedding_uses_current_time_when_endpoint_is_omitted(self):
        common = dict(
            image_features_type="CLIPMiddle_layer4",
            latent_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_mode="no_cond",
            arch="trans_enc",
            normalize_model_output=0,
        )
        baseline = SwiftSketch(diffusion_mode="ddpm", **common).eval()
        cumulative = SwiftSketch(diffusion_mode="cfm_ddim", **common).eval()
        missing, unexpected = cumulative.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertTrue(missing)
        self.assertTrue(
            all(key.startswith("embed_endpoint_timestep.") for key in missing)
        )
        self.assertFalse(unexpected)
        cumulative.initialize_endpoint_timestep()

        x = torch.randn(2, 3, 4, 2)
        cfm_time = torch.tensor([0.1, 0.5])
        with torch.no_grad():
            default_endpoint_output = cumulative(x, cfm_time)
            explicit_endpoint_output = cumulative(
                x, cfm_time, end_timesteps=cfm_time
            )
        self.assertTrue(
            torch.allclose(
                default_endpoint_output,
                explicit_endpoint_output,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_jvp_loss_has_finite_parameter_gradients(self):
        diffusion = make_diffusion()
        model = CountingModel()
        x_0 = torch.randn(2, 3, 4, 2)
        time_t = torch.tensor([0.65, 0.5])
        time_r = torch.tensor([0.2, 0.5])
        terms = diffusion.training_cfm_ddim_losses(
            model,
            x_0,
            image_features=None,
            time_t=time_t,
            time_r=time_r,
        )
        terms["loss"].mean().backward()
        self.assertTrue(torch.isfinite(terms["loss"]).all())
        self.assertIsNotNone(model.weight.grad)
        self.assertTrue(torch.isfinite(model.weight.grad))

    def test_full_transformer_jvp_smoke(self):
        diffusion = make_normalized_cosine_diffusion(50)
        model = SwiftSketch(
            diffusion_mode="cfm_ddim",
            image_features_type="CLIPMiddle_layer4",
            latent_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_mode="no_cond",
            arch="trans_enc",
            normalize_model_output=0,
        )
        x_0 = torch.randn(2, 3, 4, 2)
        terms = diffusion.training_cfm_ddim_losses(
            model,
            x_0,
            image_features=None,
            time_t=torch.tensor([0.7, 0.5]),
            time_r=torch.tensor([0.2, 0.5]),
        )
        terms["loss"].mean().backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(
                torch.isfinite(gradient).all()
                for gradient in gradients
                if gradient is not None
            )
        )

    def test_conditioned_decoder_jvp_smoke(self):
        diffusion = make_diffusion()
        model = SwiftSketch(
            diffusion_mode="cfm_ddim",
            image_features_type="CLIPMiddle_layer4",
            latent_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_mode="image",
            cond_mask_prob=0.1,
            arch="trans_dec",
            normalize_model_output=0,
        )
        x_0 = torch.randn(1, 3, 4, 2)
        image_features = torch.randn(1, 1024, 14, 14)
        terms = diffusion.training_cfm_ddim_losses(
            model,
            x_0,
            image_features=image_features,
            time_t=torch.tensor([0.6]),
            time_r=torch.tensor([0.2]),
        )
        terms["loss"].mean().backward()
        self.assertTrue(torch.isfinite(terms["loss"]).all())
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

    def test_sampler_uses_exact_requested_call_count(self):
        diffusion = make_diffusion()
        shape = (2, 3, 4, 2)
        initial_noise = torch.randn(*shape)

        one_step_model = CountingModel()
        output = diffusion.cfm_ddim_sample_loop(
            one_step_model,
            shape,
            image_features=None,
            num_steps=1,
            noise=initial_noise,
        )
        self.assertEqual(one_step_model.calls, 1)
        self.assertEqual(tuple(output.shape), shape)

        four_step_model = CountingModel()
        diffusion.cfm_ddim_sample_loop(
            four_step_model,
            shape,
            image_features=None,
            num_steps=4,
            noise=initial_noise,
        )
        self.assertEqual(four_step_model.calls, 4)

    def test_cfg_passes_same_endpoint_to_both_branches(self):
        inner_model = RecordingConditionalModel()
        guided_model = ClassifierFreeSampleModel(inner_model)
        x = torch.randn(2, 3, 4, 2)
        time_t = torch.tensor([1.0, 1.0])
        time_r = torch.tensor([0.0, 0.0])
        guided_model(
            x,
            time_t,
            image_features=torch.empty(2, 0),
            scale=torch.full((2,), 2.5),
            end_timesteps=time_r,
        )
        self.assertEqual(len(inner_model.endpoint_records), 2)
        self.assertTrue(
            torch.equal(inner_model.endpoint_records[0][0], time_r)
        )
        self.assertTrue(
            torch.equal(inner_model.endpoint_records[1][0], time_r)
        )
        self.assertFalse(inner_model.endpoint_records[0][1])
        self.assertTrue(inner_model.endpoint_records[1][1])


if __name__ == "__main__":
    unittest.main()
