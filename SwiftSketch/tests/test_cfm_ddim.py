import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

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
from train.training_loop import (
    TrainLoop,
    calculate_cfm_validation_loss_metrics,
    expand_cfm_training_batch,
    format_validation_console_message,
    sample_cfm_time_pairs,
)
from utils.parser_util import (
    get_cfm_instantaneous_samples_per_example,
    resolve_cfm_validation_sampling,
    train_args,
)


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
    def test_time_sample_count_requires_an_exact_instantaneous_ratio(self):
        self.assertIsNone(
            get_cfm_instantaneous_samples_per_example(1, 0.5)
        )
        self.assertEqual(
            get_cfm_instantaneous_samples_per_example(4, 0.5),
            2,
        )
        self.assertEqual(
            get_cfm_instantaneous_samples_per_example(4, 0.25),
            1,
        )
        with self.assertRaisesRegex(ValueError, "cannot represent"):
            get_cfm_instantaneous_samples_per_example(3, 0.5)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            get_cfm_instantaneous_samples_per_example(0, 0.5)

    def test_training_parser_rejects_invalid_cfm_validation_but_not_ddpm(self):
        common_arguments = [
            "train",
            "--save_dir",
            "unused",
            "--cfm_time_samples_per_example",
            "4",
            "--cfm_instantaneous_prob",
            "0.5",
            "--val_data_dir",
            "unused",
            "--cfm_val_time_samples_per_example",
            "1",
            "--cfm_val_instantaneous_prob",
            "0.5",
        ]
        with patch("sys.argv", common_arguments + ["--diffusion_mode", "ddpm"]):
            ddpm_args = train_args()
        self.assertEqual(ddpm_args.cfm_val_time_samples_per_example, 1)

        with patch(
            "sys.argv",
            common_arguments
            + [
                "--diffusion_mode",
                "cfm_ddim",
                "--lpips_weight",
                "0",
                "--l1_points_weight",
                "0",
            ],
        ):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    train_args()

    def test_validation_sampling_inherits_and_validates_effective_settings(self):
        self.assertEqual(
            resolve_cfm_validation_sampling(4, 0.5, 0, None),
            (4, 0.5),
        )
        self.assertEqual(
            resolve_cfm_validation_sampling(4, 0.5, 8, 0.25),
            (8, 0.25),
        )
        self.assertEqual(
            resolve_cfm_validation_sampling(1, 0.5, 1, 0.0),
            (1, 0.0),
        )
        self.assertEqual(
            resolve_cfm_validation_sampling(1, 0.5, 1, 1.0),
            (1, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "must be at least 2"):
            resolve_cfm_validation_sampling(1, 0.5, 0, None)
        with self.assertRaisesRegex(ValueError, "cannot represent"):
            resolve_cfm_validation_sampling(4, 0.5, 3, 0.5)

    def test_cfm_validation_rules_apply_only_when_validation_is_enabled(self):
        cfm_arguments = [
            "train",
            "--save_dir",
            "unused",
            "--diffusion_mode",
            "cfm_ddim",
            "--lpips_weight",
            "0",
            "--l1_points_weight",
            "0",
        ]
        with patch("sys.argv", cfm_arguments):
            args_without_validation = train_args()
        self.assertEqual(args_without_validation.cfm_time_samples_per_example, 1)

        with patch("sys.argv", cfm_arguments + ["--val_data_dir", "unused"]):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    train_args()

    def test_cfm_validation_metrics_are_weighted_and_balanced(self):
        metrics = calculate_cfm_validation_loss_metrics(
            instantaneous_loss_total=4.0,
            instantaneous_count=2,
            cumulative_loss_total=12.0,
            cumulative_count=3,
            instantaneous_probability=0.25,
        )
        self.assertEqual(metrics["Validation/loss_instantaneous"], 2.0)
        self.assertEqual(metrics["Validation/loss_cumulative"], 4.0)
        self.assertEqual(metrics["Validation/loss"], 3.5)
        self.assertEqual(metrics["Validation/loss_balanced"], 3.0)

        equal_weight_metrics = calculate_cfm_validation_loss_metrics(
            instantaneous_loss_total=4.0,
            instantaneous_count=2,
            cumulative_loss_total=12.0,
            cumulative_count=3,
            instantaneous_probability=0.5,
        )
        self.assertEqual(
            equal_weight_metrics["Validation/loss"],
            equal_weight_metrics["Validation/loss_balanced"],
        )

    def test_cfm_validation_metrics_omit_unavailable_regimes(self):
        instantaneous_only = calculate_cfm_validation_loss_metrics(
            instantaneous_loss_total=6.0,
            instantaneous_count=2,
            cumulative_loss_total=0.0,
            cumulative_count=0,
            instantaneous_probability=1.0,
        )
        self.assertEqual(
            instantaneous_only,
            {
                "Validation/loss_instantaneous": 3.0,
                "Validation/loss": 3.0,
            },
        )

        cumulative_only = calculate_cfm_validation_loss_metrics(
            instantaneous_loss_total=0.0,
            instantaneous_count=0,
            cumulative_loss_total=8.0,
            cumulative_count=2,
            instantaneous_probability=0.0,
        )
        self.assertEqual(
            cumulative_only,
            {
                "Validation/loss_cumulative": 4.0,
                "Validation/loss": 4.0,
            },
        )

    def test_validation_console_output_reports_cfm_split_metrics(self):
        message = format_validation_console_message(
            10,
            {
                "Validation/loss_instantaneous": 2.0,
                "Validation/loss_cumulative": 4.0,
                "Validation/loss_balanced": 3.0,
                "Validation/loss": 3.5,
            },
            "cfm_ddim",
        )
        self.assertEqual(
            message,
            "step[10]: val_instantaneous[2.00000] val_cumulative[4.00000] "
            "val_balanced[3.00000] val_weighted[3.50000]",
        )

    def test_ddpm_validation_console_output_is_unchanged(self):
        message = format_validation_console_message(
            10,
            {"Validation/loss": 5.0},
            "ddpm",
        )
        self.assertEqual(message, "step[10]: validation_loss[5.00000]")

    def test_validation_loop_uses_validation_sampling_and_split_metrics(self):
        class ValidationLossDiffusion:
            def __init__(self):
                self.received_batch_size = None

            def training_cfm_ddim_losses(
                inner_self,
                model,
                target_control_points,
                image_features,
                time_t,
                time_r,
                noise,
            ):
                inner_self.received_batch_size = target_control_points.shape[0]
                loss = torch.where(
                    time_t == time_r,
                    torch.tensor(2.0),
                    torch.tensor(4.0),
                )
                return {"loss": loss}

        loop = TrainLoop.__new__(TrainLoop)
        loop.model = CountingModel()
        loop.model.train()
        loop.diffusion = ValidationLossDiffusion()
        loop.validation_data = [
            (
                torch.randn(2, 3, 4, 2),
                torch.empty(2, 1, 1, 3),
                torch.randn(2, 5),
            )
        ]
        loop.diffusion_mode = "cfm_ddim"
        loop.device = torch.device("cpu")
        loop.val_max_batches = 0
        loop.val_seed = 1234
        loop.cfm_val_time_samples_per_example = 4
        loop.cfm_val_instantaneous_prob = 0.25
        loop.resume_step = 0
        loop.args = SimpleNamespace(use_wandb=0)

        with redirect_stdout(StringIO()):
            metrics = loop.evaluate_validation(global_step=10)

        self.assertEqual(loop.diffusion.received_batch_size, 8)
        self.assertEqual(metrics["Validation/loss_instantaneous"], 2.0)
        self.assertEqual(metrics["Validation/loss_cumulative"], 4.0)
        self.assertEqual(metrics["Validation/loss"], 3.5)
        self.assertEqual(metrics["Validation/loss_balanced"], 3.0)
        self.assertTrue(loop.model.training)

    def test_ddpm_validation_path_does_not_emit_cfm_split_metrics(self):
        class DDPMValidationDiffusion:
            num_timesteps = 8

            def __init__(self):
                self.received_rendered_shape = None

            def training_losses(
                inner_self,
                model,
                target_control_points,
                target_rendered_images,
                image_features,
                timesteps,
                global_step,
                resume_step,
                noise,
                mode,
                log_results,
            ):
                inner_self.received_rendered_shape = tuple(
                    target_rendered_images.shape
                )
                return {"loss": torch.full((target_control_points.shape[0],), 5.0)}

        loop = TrainLoop.__new__(TrainLoop)
        loop.model = CountingModel()
        loop.model.train()
        loop.diffusion = DDPMValidationDiffusion()
        loop.validation_data = [
            (
                torch.randn(2, 3, 4, 2),
                torch.randn(2, 4, 4, 3),
                torch.randn(2, 5),
            )
        ]
        loop.diffusion_mode = "ddpm"
        loop.device = torch.device("cpu")
        loop.val_max_batches = 0
        loop.val_seed = 1234
        loop.resume_step = 0
        loop.args = SimpleNamespace(use_wandb=0)

        with redirect_stdout(StringIO()):
            metrics = loop.evaluate_validation(global_step=10)

        self.assertEqual(loop.diffusion.received_rendered_shape, (2, 3, 4, 4))
        self.assertEqual(metrics["Validation/loss"], 5.0)
        self.assertNotIn("Validation/loss_instantaneous", metrics)
        self.assertNotIn("Validation/loss_cumulative", metrics)
        self.assertNotIn("Validation/loss_balanced", metrics)
        self.assertTrue(loop.model.training)

    def test_multi_time_sampling_has_exact_ratio_for_every_example(self):
        batch_size = 3
        samples_per_example = 4
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        time_t, time_r = sample_cfm_time_pairs(
            batch_size,
            maximum_time=1.0,
            instantaneous_probability=0.5,
            device=torch.device("cpu"),
            generator=generator,
            samples_per_example=samples_per_example,
        )

        self.assertEqual(tuple(time_t.shape), (batch_size * samples_per_example,))
        self.assertEqual(tuple(time_r.shape), (batch_size * samples_per_example,))
        self.assertTrue(torch.all(time_r <= time_t))
        instantaneous = torch.isclose(time_t, time_r).reshape(
            batch_size, samples_per_example
        )
        self.assertTrue(
            torch.equal(
                instantaneous.sum(dim=1),
                torch.full((batch_size,), 2, dtype=torch.long),
            )
        )

    def test_single_time_sample_preserves_legacy_bernoulli_behavior(self):
        generator = torch.Generator(device="cpu").manual_seed(23)
        time_t, time_r = sample_cfm_time_pairs(
            5,
            maximum_time=1.0,
            instantaneous_probability=0.5,
            device=torch.device("cpu"),
            generator=generator,
        )

        legacy_generator = torch.Generator(device="cpu").manual_seed(23)
        legacy_random_times = torch.rand(
            5, 2, generator=legacy_generator, dtype=torch.float32
        )
        legacy_random_times = 1e-3 + legacy_random_times * (1.0 - 1e-3)
        legacy_time_t = legacy_random_times.max(dim=1).values
        legacy_time_r = legacy_random_times.min(dim=1).values
        legacy_instantaneous = (
            torch.rand(5, generator=legacy_generator) < 0.5
        )
        legacy_time_r = torch.where(
            legacy_instantaneous, legacy_time_t, legacy_time_r
        )

        self.assertTrue(torch.equal(time_t, legacy_time_t))
        self.assertTrue(torch.equal(time_r, legacy_time_r))

    def test_single_validation_regime_is_deterministic_at_probability_endpoints(self):
        cumulative_t, cumulative_r = sample_cfm_time_pairs(
            128,
            maximum_time=1.0,
            instantaneous_probability=0.0,
            device=torch.device("cpu"),
        )
        self.assertFalse(torch.isclose(cumulative_t, cumulative_r).any())

        instantaneous_t, instantaneous_r = sample_cfm_time_pairs(
            128,
            maximum_time=1.0,
            instantaneous_probability=1.0,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(instantaneous_t, instantaneous_r))

    def test_cfm_batch_expansion_is_flattened_and_batch_aligned(self):
        targets = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
        features = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        expanded_targets, expanded_features = expand_cfm_training_batch(
            targets,
            features,
            samples_per_example=4,
        )

        self.assertEqual(tuple(expanded_targets.shape), (8, 3, 2))
        self.assertEqual(tuple(expanded_features.shape), (8, 2))
        for original_index in range(2):
            start = original_index * 4
            end = start + 4
            self.assertTrue(
                torch.equal(
                    expanded_targets[start:end],
                    targets[original_index].unsqueeze(0).expand(4, -1, -1),
                )
            )
            self.assertTrue(
                torch.equal(
                    expanded_features[start:end],
                    features[original_index].unsqueeze(0).expand(4, -1),
                )
            )

    def test_expanded_cfm_batch_runs_as_one_parallel_model_call(self):
        diffusion = make_diffusion()
        model = CountingModel()
        targets = torch.randn(2, 3, 4, 2)
        expanded_targets, _ = expand_cfm_training_batch(targets, None, 4)
        time_t, time_r = sample_cfm_time_pairs(
            batch_size=2,
            maximum_time=1.0,
            instantaneous_probability=0.5,
            device=targets.device,
            samples_per_example=4,
        )

        terms = diffusion.training_cfm_ddim_losses(
            model,
            expanded_targets,
            image_features=None,
            time_t=time_t,
            time_r=time_r,
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(tuple(terms["loss"].shape), (8,))
        terms["loss"].mean().backward()
        self.assertIsNotNone(model.weight.grad)

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
