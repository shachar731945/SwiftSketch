import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import call, mock_open, patch

import numpy as np
import torch
from torch import nn

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
    sampling_snapshot_indices,
    use_pydiffvg_device,
)
from model.SwiftSketch_model import SwiftSketch
from model.cfg_sampler import ClassifierFreeSampleModel
from train.training_loop import (
    TrainLoop,
    calculate_cfm_lpips_validation_metrics,
    calculate_cfm_validation_loss_metrics,
    expand_cfm_training_batch,
    format_validation_console_message,
    log_loss_dict,
    sample_cfm_time_pairs,
)
from train.validation_media import (
    _final_comparison_tiles,
    _render_normalized_points,
    _resolve_render_device,
    _use_pydiffvg_device,
    _wandb_tiles,
    generate_cfm_media_tensors,
)
from utils.model_util import get_model_args
from utils.parser_util import (
    CFM_TIME_VERSION,
    get_cfm_instantaneous_samples_per_example,
    resolve_cfm_validation_sampling,
    train_args,
)


def make_diffusion(num_steps=8, lpips_weight=0.0):
    args = SimpleNamespace(
        diffusion_mode="cfm_ddim",
        cfm_loss_weight=1.0,
        lpips_weight=lpips_weight,
        device="cpu",
        scaling_factor=1.0,
        canvas_width=32,
        canvas_height=32,
    )
    return GaussianDiffusion(
        args=args,
        betas=np.linspace(1e-3, 2e-2, num_steps),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
    )


def make_ddpm_diffusion(num_steps=4):
    args = SimpleNamespace(
        diffusion_mode="ddpm",
        use_wandb=0,
        diffusion_steps=num_steps,
        generate=1,
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
        timesteps=None,
        image_features=None,
        end_timesteps=None,
        scale=None,
        ts=None,
    ):
        self.calls += 1
        if timesteps is None:
            timesteps = ts
        time = timesteps.reshape(-1, *([1] * (x.ndim - 1)))
        if end_timesteps is None:
            end_timesteps = timesteps
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
    def test_cfm_lpips_parser_accepts_only_the_supported_configuration(self):
        valid_arguments = [
            "train",
            "--save_dir",
            "unused",
            "--diffusion_mode",
            "cfm_ddim",
            "--lpips_weight",
            "0.2",
            "--l1_points_weight",
            "0",
            "--cfm_time_samples_per_example",
            "1",
            "--cfm_val_time_samples_per_example",
            "1",
            "--normalize_model_output",
            "1",
            "--val_data_dir",
            "unused",
        ]
        with patch("sys.argv", valid_arguments):
            args = train_args()

        self.assertEqual(args.lpips_weight, 0.2)
        self.assertEqual(args.l1_points_weight, 0.0)
        self.assertEqual(args.cfm_time_samples_per_example, 1)
        self.assertEqual(args.cfm_val_time_samples_per_example, 1)
        self.assertEqual(args.normalize_model_output, 1)

        invalid_suffixes = (
            ["--cfm_time_samples_per_example", "2"],
            ["--cfm_val_time_samples_per_example", "2"],
            ["--normalize_model_output", "0"],
            ["--l1_points_weight", "1"],
        )
        for suffix in invalid_suffixes:
            with self.subTest(suffix=suffix), patch(
                "sys.argv", valid_arguments + suffix
            ), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    train_args()

    def test_cfm_loads_lpips_only_when_its_weight_is_positive(self):
        with patch("diffusion.gaussian_diffusion.LPIPS") as lpips_constructor:
            pure_cfm = make_diffusion(lpips_weight=0.0)
            self.assertIsNone(pure_cfm.cfm_lpips_func)
            lpips_constructor.assert_not_called()

            perceptual_cfm = make_diffusion(lpips_weight=0.2)
            self.assertIs(perceptual_cfm.cfm_lpips_func, lpips_constructor.return_value)
            lpips_constructor.assert_called_once()

    def test_wandb_media_omits_noise_and_has_the_three_final_comparison_tiles(self):
        tiles = [
            ("Input", None, None, "input.png"),
            ("Ground truth", None, None, "ground_truth.svg"),
            ("Fixed initial noise", None, None, "initial_noise.svg"),
            ("CFM 1-step final", None, None, "cfm_1_step_final.svg"),
            ("CFM 4-step final", None, None, "cfm_4_step_final.svg"),
            (
                "Step 3 prediction",
                None,
                None,
                "cfm_4_step_intermediates/step_0003_of_0004_prediction.svg",
            ),
            (
                "Step 4 state",
                None,
                None,
                "cfm_4_step_intermediates/step_0004_of_0004_state.svg",
            ),
            (
                "Step 4 prediction",
                None,
                None,
                "cfm_4_step_intermediates/step_0004_of_0004_prediction.svg",
            ),
            ("Other", None, None, "other.svg"),
        ]

        self.assertEqual(
            [tile[3] for tile in _wandb_tiles(tiles)],
            [
                "input.png",
                "ground_truth.svg",
                "cfm_1_step_final.svg",
                "cfm_4_step_final.svg",
                "cfm_4_step_intermediates/step_0003_of_0004_prediction.svg",
                "cfm_4_step_intermediates/step_0004_of_0004_state.svg",
                "other.svg",
            ],
        )
        self.assertEqual(
            [tile[3] for tile in _final_comparison_tiles(tiles)],
            [
                "ground_truth.svg",
                "cfm_1_step_final.svg",
                "cfm_4_step_final.svg",
            ],
        )

    def test_cfm_parser_records_current_time_conditioning_version(self):
        with patch(
            "sys.argv",
            [
                "train",
                "--save_dir",
                "unused",
                "--diffusion_mode",
                "cfm_ddim",
                "--diffusion_steps",
                "50",
                "--lpips_weight",
                "0",
                "--l1_points_weight",
                "0",
            ],
        ):
            args = train_args()

        self.assertEqual(args.cfm_time_version, CFM_TIME_VERSION)
        self.assertEqual(get_model_args(args)["cfm_time_embedding_scale"], 49)

    def test_checkpoint_version_rejects_old_cfm_but_accepts_ddpm_initialization(self):
        loop = TrainLoop.__new__(TrainLoop)
        loop.diffusion_mode = "cfm_ddim"

        with patch("train.training_loop.bf.exists", return_value=True), patch(
            "train.training_loop.bf.BlobFile",
            mock_open(
                read_data=json.dumps(
                    {"diffusion_mode": "cfm_ddim", "cfm_time_version": 2}
                )
            ),
        ):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                loop._validate_cfm_checkpoint_version("old/model.pt", "resume")

        with patch("train.training_loop.bf.exists", return_value=True), patch(
            "train.training_loop.bf.BlobFile",
            mock_open(
                read_data=json.dumps(
                    {
                        "diffusion_mode": "cfm_ddim",
                        "cfm_time_version": CFM_TIME_VERSION,
                    }
                )
            ),
        ):
            loop._validate_cfm_checkpoint_version("current/model.pt", "resume")

        with patch("train.training_loop.bf.exists", return_value=True), patch(
            "train.training_loop.bf.BlobFile",
            mock_open(read_data=json.dumps({"diffusion_mode": "ddpm"})),
        ):
            loop._validate_cfm_checkpoint_version("ddpm/model.pt", "initialize")

    def test_cfm_embedding_coordinates_match_the_ddpm_grid(self):
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
        ddpm = SwiftSketch(diffusion_mode="ddpm", **common)
        cfm = SwiftSketch(
            diffusion_mode="cfm_ddim",
            cfm_time_embedding_scale=49,
            **common,
        )
        ddpm_indices = torch.tensor([0.0, 1.0, 17.0, 49.0])
        normalized_times = ddpm_indices / 49.0

        self.assertTrue(
            torch.equal(
                cfm.timestep_embedding_coordinate(normalized_times),
                ddpm.timestep_embedding_coordinate(ddpm_indices),
            )
        )

    def test_cfm_forward_scales_both_time_embedding_coordinates(self):
        model = SwiftSketch(
            diffusion_mode="cfm_ddim",
            cfm_time_embedding_scale=49,
            image_features_type="CLIPMiddle_layer4",
            latent_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_mode="no_cond",
            arch="trans_enc",
            normalize_model_output=0,
        ).eval()
        current_times = torch.tensor([0.25, 1.0])
        endpoint_times = torch.tensor([0.1, 0.5])

        with patch.object(
            model.embed_timestep,
            "forward",
            wraps=model.embed_timestep.forward,
        ) as current_embedder, patch.object(
            model.embed_endpoint_timestep,
            "forward",
            wraps=model.embed_endpoint_timestep.forward,
        ) as endpoint_embedder, torch.no_grad():
            model(
                torch.randn(2, 3, 4, 2),
                current_times,
                end_timesteps=endpoint_times,
            )

        self.assertTrue(
            torch.equal(current_embedder.call_args.args[0], current_times * 49)
        )
        self.assertTrue(
            torch.equal(endpoint_embedder.call_args.args[0], endpoint_times * 49)
        )

    def test_ddpm_forward_ignores_cfm_embedding_scale(self):
        common = dict(
            diffusion_mode="ddpm",
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
        baseline = SwiftSketch(cfm_time_embedding_scale=1, **common).eval()
        changed_scale = SwiftSketch(cfm_time_embedding_scale=100, **common).eval()
        changed_scale.load_state_dict(baseline.state_dict())
        x = torch.randn(2, 3, 4, 2)
        timesteps = torch.tensor([3, 17])

        with torch.no_grad():
            baseline_output = baseline(x, timesteps)
            changed_scale_output = changed_scale(x, timesteps)

        self.assertTrue(torch.equal(baseline_output, changed_scale_output))

    def test_unscaled_model_output_is_not_bounded_by_the_data_scale(self):
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
            scaling_factor=2,
        ).eval()
        with torch.no_grad():
            model.output_process.pointsFinal.weight.zero_()
            model.output_process.pointsFinal.bias.fill_(3.0)
            output = model(torch.randn(1, 3, 4, 2), torch.tensor([0.5]))

        self.assertTrue(torch.equal(output, torch.full_like(output, 3.0)))

    def test_normalized_model_output_uses_existing_tanh_data_scale(self):
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
            normalize_model_output=1,
            scaling_factor=2,
        ).eval()
        with torch.no_grad():
            model.output_process.pointsFinal.weight.zero_()
            model.output_process.pointsFinal.bias.fill_(100.0)
            output = model(torch.randn(1, 3, 4, 2), torch.tensor([0.5]))

        self.assertTrue(torch.equal(output, torch.full_like(output, 2.0)))

    def test_training_media_is_opt_in_and_rejected_for_ddpm(self):
        with patch(
            "sys.argv",
            ["train", "--save_dir", "unused", "--diffusion_mode", "ddpm"],
        ):
            ddpm_args = train_args()
        self.assertEqual(ddpm_args.media_interval, 0)

        with patch(
            "sys.argv",
            [
                "train",
                "--save_dir",
                "unused",
                "--diffusion_mode",
                "ddpm",
                "--media_interval",
                "100",
                "--media_validation_files",
                "example.npz",
            ],
        ):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    train_args()

    def test_training_media_parser_accepts_cfm_defaults_and_checks_step_count(self):
        valid_arguments = [
            "train",
            "--save_dir",
            "unused",
            "--diffusion_mode",
            "cfm_ddim",
            "--diffusion_steps",
            "4",
            "--lpips_weight",
            "0",
            "--l1_points_weight",
            "0",
            "--media_interval",
            "100",
            "--media_validation_files",
            "example.npz",
        ]
        with patch("sys.argv", valid_arguments):
            args = train_args()
        self.assertEqual(args.media_cfm_sampling_steps, [1, 4])
        self.assertEqual(args.media_instantaneous_times, [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(args.media_output_mode, "both")
        self.assertEqual(args.validation_media_render_device, "cpu")

        with patch(
            "sys.argv",
            valid_arguments + ["--validation_media_render_device", "model"],
        ):
            model_device_args = train_args()
        self.assertEqual(model_device_args.validation_media_render_device, "model")

        with patch(
            "sys.argv",
            valid_arguments
            + ["--media_cfm_sampling_steps", "1", "5"],
        ):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    train_args()

        with patch(
            "sys.argv",
            valid_arguments + ["--validation_media_render_device", "cuda"],
        ):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    train_args()

    def test_validation_media_render_device_is_resolved_and_scoped(self):
        self.assertEqual(
            _resolve_render_device("cpu", torch.device("cuda:1")),
            torch.device("cpu"),
        )
        self.assertEqual(
            _resolve_render_device("model", torch.device("cuda:1")),
            torch.device("cuda:1"),
        )
        with self.assertRaisesRegex(ValueError, "either 'cpu' or 'model'"):
            _resolve_render_device("cuda", torch.device("cuda:1"))

        previous_device = torch.device("cuda:0")
        with patch(
            "train.validation_media.pydiffvg.get_device",
            return_value=previous_device,
        ), patch("train.validation_media.pydiffvg.set_device") as set_device:
            with _use_pydiffvg_device(torch.device("cpu")):
                pass

        self.assertEqual(
            set_device.call_args_list,
            [
                call(torch.device("cpu")),
                call(previous_device),
            ],
        )

    def test_validation_media_cpu_render_contains_visible_strokes(self):
        canvas_size = 32
        canvas_points = torch.tensor(
            [[[[2.0, 2.0], [10.0, 2.0], [22.0, 30.0], [30.0, 30.0]]]]
        )
        normalized_points = (canvas_points / canvas_size * 2.0) - 1.0
        args = SimpleNamespace(
            scaling_factor=1.0,
            canvas_width=canvas_size,
            canvas_height=canvas_size,
        )

        with _use_pydiffvg_device(torch.device("cpu")):
            image, svg = _render_normalized_points(normalized_points, args)

        self.assertLess(np.asarray(image).min(), 250)
        self.assertIn("<path", svg)

    def test_cfm_media_generation_is_fixed_and_contains_requested_views(self):
        diffusion = make_diffusion(num_steps=4)
        model = CountingModel().eval()
        target = torch.randn(3, 4, 2)
        features = torch.randn(5)

        torch.manual_seed(91)
        expected_next_random_value = torch.rand(1)
        torch.manual_seed(91)
        first = generate_cfm_media_tensors(
            model,
            diffusion,
            target,
            features,
            torch.device("cpu"),
            seed=17,
            guidance_param=1.0,
            sampling_steps=[1, 4],
            instantaneous_times=[0.25, 0.5, 0.75, 1.0],
        )
        actual_next_random_value = torch.rand(1)
        second = generate_cfm_media_tensors(
            model,
            diffusion,
            target,
            features,
            torch.device("cpu"),
            seed=17,
            guidance_param=1.0,
            sampling_steps=[1, 4],
            instantaneous_times=[0.25, 0.5, 0.75, 1.0],
        )

        self.assertTrue(torch.equal(actual_next_random_value, expected_next_random_value))
        self.assertEqual(set(first["final_samples"]), {1, 4})
        self.assertEqual(first["detailed_step_count"], 4)
        self.assertEqual(len(first["intermediates"]), 4)
        self.assertEqual(set(first["instantaneous_predictions"]), {0.25, 0.5, 0.75, 1.0})
        self.assertTrue(torch.equal(first["initial_noise"], second["initial_noise"]))
        self.assertTrue(
            torch.equal(first["final_samples"][4], second["final_samples"][4])
        )
        self.assertTrue(
            torch.equal(
                first["final_samples"][4],
                first["intermediates"][-1]["state"],
            )
        )

    def test_sampling_snapshot_indices_include_final_without_initial_noise(self):
        self.assertEqual(sampling_snapshot_indices(4, 4), {0, 1, 2, 3})
        self.assertEqual(sampling_snapshot_indices(10, 4), {2, 4, 7, 9})
        self.assertEqual(sampling_snapshot_indices(10, 1), {9})
        self.assertEqual(sampling_snapshot_indices(10, 0), set())
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            sampling_snapshot_indices(4, 5)

    def test_cfm_sampling_returns_requested_state_prediction_pairs(self):
        diffusion = make_diffusion(num_steps=4)
        model = CountingModel()
        final, intermediates = diffusion.cfm_ddim_sample_loop(
            model,
            shape=(2, 3, 4, 2),
            image_features=None,
            num_steps=4,
            noise=torch.zeros(2, 3, 4, 2),
            return_intermediates=True,
            intermediate_steps=4,
        )
        self.assertEqual(model.calls, 4)
        self.assertEqual(len(intermediates), 4)
        self.assertEqual([item["step"] for item in intermediates], [1, 2, 3, 4])
        self.assertTrue(torch.equal(final.cpu(), intermediates[-1]["state"]))
        for item in intermediates:
            self.assertEqual(tuple(item["state"].shape), (2, 3, 4, 2))
            self.assertEqual(tuple(item["prediction"].shape), (2, 3, 4, 2))
            self.assertEqual(item["state"].device.type, "cpu")

    def test_cfm_sampler_infers_device_from_swiftsketch_parameter_list(self):
        diffusion = make_diffusion(num_steps=2)
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
        ).eval()

        output = diffusion.cfm_ddim_sample_loop(
            model,
            shape=(1, 3, 4, 2),
            image_features=None,
            num_steps=1,
        )

        self.assertEqual(tuple(output.shape), (1, 3, 4, 2))

    def test_ddpm_sampling_returns_requested_state_prediction_pairs(self):
        diffusion = make_ddpm_diffusion(num_steps=4)
        model = CountingModel()
        final, intermediates = diffusion.p_sample_loop(
            model,
            shape=(2, 3, 4, 2),
            noise=torch.zeros(2, 3, 4, 2),
            clip_denoised=False,
            return_intermediates=True,
            intermediate_steps=4,
        )
        self.assertEqual(model.calls, 4)
        self.assertEqual(len(intermediates), 4)
        self.assertEqual([item["step"] for item in intermediates], [1, 2, 3, 4])
        self.assertTrue(torch.equal(final.cpu(), intermediates[-1]["state"]))
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

    def test_cfm_lpips_validation_metrics_include_raw_weighted_and_balanced_losses(self):
        metrics = calculate_cfm_lpips_validation_metrics(
            cfm_instantaneous_total=4.0,
            cfm_cumulative_total=8.0,
            lpips_instantaneous_total=6.0,
            lpips_terminal_total=10.0,
            num_examples=2,
            instantaneous_probability=0.25,
            cfm_weight=1.0,
            lpips_weight=0.2,
        )

        self.assertEqual(metrics["Validation/cfm_instantaneous_loss"], 2.0)
        self.assertEqual(metrics["Validation/cfm_cumulative_loss"], 4.0)
        self.assertEqual(metrics["Validation/lpips_instantaneous_loss"], 3.0)
        self.assertEqual(metrics["Validation/lpips_terminal_loss"], 5.0)
        self.assertEqual(metrics["Validation/loss"], 4.4)
        self.assertEqual(metrics["Validation/loss_balanced"], 3.8)

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
            "step[10]: val_instantaneous_loss[2.00000] "
            "val_cumulative_loss[4.00000] val_balanced_loss[3.00000] "
            "val_weighted_loss[3.50000]",
        )

        lpips_message = format_validation_console_message(
            10,
            {
                "Validation/cfm_instantaneous_loss": 2.0,
                "Validation/cfm_cumulative_loss": 4.0,
                "Validation/lpips_instantaneous_loss": 3.0,
                "Validation/lpips_terminal_loss": 5.0,
                "Validation/loss_balanced": 3.8,
                "Validation/loss": 4.4,
            },
            "cfm_ddim",
        )
        self.assertEqual(
            lpips_message,
            "step[10]: val_cfm_instantaneous_loss[2.00000] "
            "val_cfm_cumulative_loss[4.00000] "
            "val_lpips_instantaneous_loss[3.00000] "
            "val_lpips_terminal_loss[5.00000] val_balanced_loss[3.80000] "
            "val_weighted_loss[4.40000]",
        )

    def test_cfm_training_logging_omits_sampling_only_diagnostics(self):
        losses = {
            "cfm_ddim_loss": torch.tensor([2.0]),
            "lpips_loss": torch.tensor(3.0),
            "loss": torch.tensor([2.5]),
            "cfm_k_abs": torch.tensor([4.0]),
            "cfm_d_rms": torch.tensor([5.0]),
            "cfm_time_coefficient_abs": torch.tensor([6.0]),
        }

        with patch("train.training_loop.logger.logkv_mean") as logkv_mean:
            log_loss_dict(None, None, losses)

        self.assertEqual(
            logkv_mean.call_args_list,
            [
                call("cfm_ddim_loss", 2.0),
                call("lpips_loss", 3.0),
                call("loss", 2.5),
                call("cfm_d_rms", 5.0),
            ],
        )

    def test_cfm_lpips_validation_evaluates_every_regime_on_cpu(self):
        class ValidationLossDiffusion:
            def __init__(self):
                self.calls = []

            def training_cfm_ddim_losses(
                inner_self,
                model,
                target_control_points,
                image_features,
                time_t,
                time_r,
                noise,
                target_rendered_images,
                mode,
                render_device,
            ):
                instantaneous = torch.equal(time_t, time_r)
                inner_self.calls.append(
                    (instantaneous, mode, render_device, target_rendered_images.shape)
                )
                batch_size = target_control_points.shape[0]
                return {
                    "cfm_ddim_loss": torch.full(
                        (batch_size,), 2.0 if instantaneous else 4.0
                    ),
                    "lpips_loss": torch.tensor(3.0 if instantaneous else 5.0),
                    "loss": torch.zeros(batch_size),
                    "cfm_k_abs": torch.zeros(batch_size),
                    "cfm_d_rms": torch.zeros(batch_size),
                    "cfm_time_coefficient_abs": torch.zeros(batch_size),
                }

        loop = TrainLoop.__new__(TrainLoop)
        loop.model = CountingModel()
        loop.model.train()
        loop.diffusion = ValidationLossDiffusion()
        loop.validation_data = [
            (
                torch.randn(2, 3, 4, 2),
                torch.randn(2, 4, 4, 3),
                torch.randn(2, 5),
            )
        ]
        loop.diffusion_mode = "cfm_ddim"
        loop.cfm_lpips_enabled = True
        loop.device = torch.device("cpu")
        loop.val_max_batches = 0
        loop.val_seed = 1234
        loop.cfm_val_time_samples_per_example = 1
        loop.cfm_val_instantaneous_prob = 0.25
        loop.resume_step = 0
        loop.args = SimpleNamespace(
            use_wandb=0,
            cfm_loss_weight=1.0,
            lpips_weight=0.2,
        )

        with redirect_stdout(StringIO()):
            metrics = loop.evaluate_validation(global_step=10)

        self.assertEqual(len(loop.diffusion.calls), 2)
        self.assertEqual(
            loop.diffusion.calls,
            [
                (True, "eval", torch.device("cpu"), torch.Size([2, 3, 4, 4])),
                (False, "eval", torch.device("cpu"), torch.Size([2, 3, 4, 4])),
            ],
        )
        self.assertEqual(metrics["Validation/cfm_instantaneous_loss"], 2.0)
        self.assertEqual(metrics["Validation/cfm_cumulative_loss"], 4.0)
        self.assertEqual(metrics["Validation/lpips_instantaneous_loss"], 3.0)
        self.assertEqual(metrics["Validation/lpips_terminal_loss"], 5.0)
        self.assertEqual(metrics["Validation/loss"], 4.4)
        self.assertEqual(metrics["Validation/loss_balanced"], 3.8)
        self.assertEqual(metrics["Validation/cfm_d_rms"], 0.0)
        self.assertNotIn("Validation/cfm_k_abs", metrics)
        self.assertNotIn("Validation/cfm_time_coefficient_abs", metrics)
        self.assertNotIn("Validation/num_examples", metrics)
        self.assertNotIn("Validation/num_batches", metrics)
        self.assertTrue(loop.model.training)

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
        self.assertTrue(torch.allclose(terms["cfm_ddim_loss"], expected))
        self.assertTrue(torch.equal(terms["cfm_k_abs"], torch.zeros(2)))

        terms["loss"].mean().backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(derivative_d.grad)

    def test_lpips_prediction_reuses_instantaneous_and_predicts_terminal_cumulative(self):
        model = CountingModel()
        x_t = torch.ones(2, 1, 4, 2)
        cfm_prediction = torch.zeros_like(x_t, requires_grad=True)
        time_t = torch.tensor([0.4, 0.8])
        time_r = torch.tensor([0.4, 0.2])

        lpips_prediction = GaussianDiffusion.select_cfm_lpips_prediction(
            model,
            cfm_prediction,
            x_t,
            time_t,
            time_r,
            image_features=None,
        )

        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.equal(lpips_prediction[0], cfm_prediction[0]))
        expected_terminal = model.weight * x_t[1] + 0.01 * time_t[1]
        self.assertTrue(torch.allclose(lpips_prediction[1], expected_terminal))

        model.calls = 0
        all_instantaneous = GaussianDiffusion.select_cfm_lpips_prediction(
            model,
            cfm_prediction,
            x_t,
            time_t,
            time_t,
            image_features=None,
        )
        self.assertEqual(model.calls, 0)
        self.assertIs(all_instantaneous, cfm_prediction)

    def test_cfm_lpips_rendering_is_differentiable_and_restores_device(self):
        class FakeLPIPS(nn.Module):
            def forward(self, prediction, target, mode="train"):
                return (prediction - target).square().mean()

        diffusion = make_diffusion()
        diffusion.cfm_lpips_func = FakeLPIPS()
        model = CountingModel()
        source = torch.ones(2, 1, 4, 2)
        times = torch.tensor([0.3, 0.6])
        prediction = model(source, times, end_timesteps=times)
        prediction.retain_grad()
        target = torch.zeros(2, 3, 4, 4, requires_grad=True)

        def fake_render(canvas_points, canvas_width, canvas_height):
            intensity = canvas_points.flatten(1).mean(1).reshape(-1, 1, 1, 1)
            return intensity.expand(-1, 4, 4, 3), [None] * canvas_points.shape[0]

        previous_device = torch.device("cuda:0")
        with patch(
            "diffusion.gaussian_diffusion.rander_image_from_points",
            side_effect=fake_render,
        ), patch(
            "diffusion.gaussian_diffusion.pydiffvg.get_device",
            return_value=previous_device,
        ), patch(
            "diffusion.gaussian_diffusion.pydiffvg.set_device"
        ) as set_device:
            loss = diffusion.calculate_cfm_lpips_loss(
                prediction,
                target,
                mode="train",
                render_device=torch.device("cuda:1"),
            )

        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNone(target.grad)
        self.assertEqual(
            set_device.call_args_list,
            [call(torch.device("cuda:1")), call(previous_device)],
        )

    def test_cfm_total_loss_adds_weighted_lpips_without_batch_scaling(self):
        diffusion = make_diffusion()
        diffusion.args.lpips_weight = 0.2
        diffusion.cfm_lpips_func = nn.Identity()
        model = CountingModel()
        x_0 = torch.randn(2, 1, 4, 2)
        time_t = torch.tensor([0.6, 0.7])
        time_r = time_t.clone()

        with patch.object(
            diffusion,
            "calculate_cfm_lpips_loss",
            return_value=torch.tensor(2.0),
        ):
            losses = diffusion.training_cfm_ddim_losses(
                model,
                x_0,
                image_features=None,
                time_t=time_t,
                time_r=time_r,
                target_rendered_images=torch.zeros(2, 3, 4, 4),
            )

        self.assertTrue(
            torch.allclose(
                losses["loss"],
                losses["cfm_ddim_loss"] + torch.tensor(0.4),
            )
        )
        self.assertEqual(losses["lpips_loss"].item(), 2.0)

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
        for key, value in cumulative.embed_timestep.state_dict().items():
            self.assertTrue(
                torch.equal(
                    value,
                    cumulative.embed_endpoint_timestep.state_dict()[key],
                )
            )

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
