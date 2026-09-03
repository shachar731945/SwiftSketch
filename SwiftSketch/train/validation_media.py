"""Deterministic validation visualizations for CFM-DDIM training.

This module deliberately stays outside the loss/optimization path.  Media is
loaded from fixed validation files, generated from a private RNG, and rendered
only when the opt-in training interval requests it.
"""

import math
import os
import re
from contextlib import contextmanager

import pydiffvg
import torch
from PIL import Image, ImageDraw

from model.cfg_sampler import ClassifierFreeSampleModel
from utils import sketch_utils
from utils.get_data import normalize_control_points


def _candidate_media_files(explicit_files, validation_directories):
    if explicit_files:
        return list(explicit_files)

    candidates = []
    for directory in validation_directories:
        if not os.path.isdir(directory):
            continue
        candidates.extend(
            os.path.join(directory, filename)
            for filename in sorted(os.listdir(directory))
            if os.path.splitext(filename)[1].lower() in {".npy", ".npz"}
        )
    return candidates


def load_fixed_media_examples(args):
    """Load explicit files, or the first valid sorted validation example."""
    explicit_files = list(getattr(args, "media_validation_files", []) or [])
    candidates = _candidate_media_files(explicit_files, args.val_data_dir)
    features_key = f"{args.image_features_type}_features"
    examples = []
    failures = []

    for file_path in candidates:
        try:
            entry = sketch_utils.load_entry(
                file_path,
                [args.target_key_name],
                features_key,
            )
            missing = {
                key
                for key in ("image", args.target_key_name, features_key)
                if key not in entry
            }
            if missing:
                raise KeyError(", ".join(sorted(missing)))

            target_svg = entry[args.target_key_name]
            points, source_canvas_size = sketch_utils.extract_control_points_from_svg(
                target_svg
            )
            if source_canvas_size is None or source_canvas_size <= 0:
                raise ValueError("target SVG has no positive canvas width")

            if args.sort_by != "no_sorting":
                mask = entry.get("mask")
                attention = entry.get("attn_map")
                points = sketch_utils.sort_strokes(
                    points,
                    args.sort_by,
                    target_svg,
                    mask,
                    attention,
                )
            points = points * (args.canvas_width / source_canvas_size)
            normalized_points = normalize_control_points(
                points,
                args.canvas_width,
                args.scaling_factor,
            )
            examples.append(
                {
                    "path": file_path,
                    "name": os.path.splitext(os.path.basename(file_path))[0],
                    "input_image": entry["image"].convert("RGB").copy(),
                    "target_points": normalized_points.float().cpu(),
                    "image_features": entry[features_key].float().cpu(),
                }
            )
        except Exception as error:
            failures.append(f"{file_path}: {error}")
            if explicit_files:
                continue

        # The automatic mode intentionally chooses one stable example.
        if examples and not explicit_files:
            break

    if not examples:
        detail = "; ".join(failures[:3]) or "no .npy/.npz candidates were found"
        raise ValueError(
            "Could not load a validation media example containing image, "
            f"{args.target_key_name}, and {features_key}. {detail}"
        )
    if explicit_files and failures:
        raise ValueError(
            "Every explicitly requested --media_validation_files entry must be valid. "
            + "; ".join(failures)
        )
    return examples


@torch.no_grad()
def generate_cfm_media_tensors(
    model,
    diffusion,
    target_points,
    image_features,
    device,
    seed,
    guidance_param,
    sampling_steps,
    instantaneous_times,
):
    """Generate fixed-noise CFM samples without consuming the training RNG."""
    target_points = target_points.unsqueeze(0).to(device)
    image_features = image_features.unsqueeze(0).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial_noise = torch.randn(
        target_points.shape,
        dtype=target_points.dtype,
        device=device,
        generator=generator,
    )

    sampling_model = model
    scale = None
    if guidance_param != 1.0:
        if getattr(model, "cond_mask_prob", 0.0) <= 0:
            raise ValueError(
                "--media_guidance_param must be 1 for a model trained without "
                "classifier-free condition masking."
            )
        sampling_model = ClassifierFreeSampleModel(model)
        scale = torch.full((1,), guidance_param, device=device)

    unique_sampling_steps = sorted(set(sampling_steps))
    detailed_step_count = max(unique_sampling_steps)
    final_samples = {}
    detailed_intermediates = []
    for step_count in unique_sampling_steps:
        wants_details = step_count == detailed_step_count
        result = diffusion.cfm_ddim_sample_loop(
            sampling_model,
            tuple(target_points.shape),
            image_features=image_features,
            num_steps=step_count,
            noise=initial_noise,
            scale=scale,
            progress=False,
            return_intermediates=wants_details,
            intermediate_steps=step_count if wants_details else 0,
        )
        if wants_details:
            sample, detailed_intermediates = result
        else:
            sample = result
        final_samples[step_count] = sample.detach().cpu()

    instantaneous_predictions = {}
    for time_value in instantaneous_times:
        times = torch.full((1,), float(time_value), device=device)
        noisy_state = diffusion.q_sample_cfm(
            target_points,
            times,
            noise=initial_noise,
        )
        prediction = sampling_model(
            x=noisy_state,
            timesteps=times,
            end_timesteps=times,
            image_features=image_features,
            scale=scale,
        )
        instantaneous_predictions[float(time_value)] = prediction.detach().cpu()

    return {
        "target": target_points.detach().cpu(),
        "initial_noise": initial_noise.detach().cpu(),
        "final_samples": final_samples,
        "detailed_step_count": detailed_step_count,
        "intermediates": detailed_intermediates,
        "instantaneous_predictions": instantaneous_predictions,
    }


def _tensor_image_to_pil(image_tensor):
    pixels = (
        image_tensor.detach().cpu().clamp(0.0, 1.0).mul(255).byte().numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def _resolve_render_device(option, model_device):
    if option == "cpu":
        return torch.device("cpu")
    if option == "model":
        return torch.device(model_device)
    raise ValueError(
        "--validation_media_render_device must be either 'cpu' or 'model'."
    )


@contextmanager
def _use_pydiffvg_device(device):
    """Temporarily select a pydiffvg device without changing training state."""
    previous_device = pydiffvg.get_device()
    pydiffvg.set_device(device)
    try:
        yield
    finally:
        pydiffvg.set_device(previous_device)


def _render_normalized_points(points, args):
    canvas_points = sketch_utils.denormalize_points(
        points.detach().cpu(),
        args.scaling_factor,
        args.canvas_width,
    )
    rendered, svg_contents = sketch_utils.rander_image_from_points(
        canvas_points,
        args.canvas_width,
        args.canvas_height,
        return_svg_content=True,
    )
    return _tensor_image_to_pil(rendered[0]), svg_contents[0]


def _time_label(time_value):
    return (f"{time_value:.4f}".rstrip("0").rstrip(".")).replace(".", "p")


def _make_overview(tiles, tile_size=224, columns=4):
    caption_height = 42
    rows = math.ceil(len(tiles) / columns)
    overview = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(overview)
    for index, (caption, image, _, _) in enumerate(tiles):
        row, column = divmod(index, columns)
        x = column * tile_size
        y = row * (tile_size + caption_height)
        thumbnail = image.convert("RGB").resize((tile_size, tile_size))
        overview.paste(thumbnail, (x, y))
        draw.multiline_text((x + 4, y + tile_size + 3), caption, fill="black")
    return overview


def _wandb_tiles(tiles):
    """Exclude reproducibility inputs and a redundant final prediction."""
    presentation_tiles = []
    final_prediction_pattern = re.compile(
        r"^step_(\d+)_of_(\d+)_prediction\.svg$"
    )
    for tile in tiles:
        relative_path = tile[3]
        if relative_path == "initial_noise.svg":
            continue
        match = final_prediction_pattern.match(os.path.basename(relative_path))
        if match is not None and match.group(1) == match.group(2):
            continue
        presentation_tiles.append(tile)
    return presentation_tiles


def _final_comparison_tiles(tiles):
    """Return the fixed three-panel final-sample comparison when available."""
    tiles_by_path = {tile[3]: tile for tile in tiles}
    required_paths = (
        "ground_truth.svg",
        "cfm_1_step_final.svg",
        "cfm_4_step_final.svg",
    )
    if not all(path in tiles_by_path for path in required_paths):
        return []
    return [tiles_by_path[path] for path in required_paths]


class CFMValidationMediaLogger:
    """Render and publish deterministic CFM validation examples."""

    def __init__(self, args, model, diffusion, device):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.examples = load_fixed_media_examples(args)
        self.seed = args.val_seed if args.media_seed is None else args.media_seed
        self.output_to_disk = args.media_output_mode in {"disk", "both"}
        self.output_to_wandb = (
            args.use_wandb and args.media_output_mode in {"wandb", "both"}
        )
        self.output_dir = args.media_output_dir or os.path.join(args.save_dir, "media")
        self.render_device = _resolve_render_device(
            getattr(args, "validation_media_render_device", "cpu"),
            device,
        )

    def log(self, global_step):
        was_training = self.model.training
        self.model.eval()
        wandb_payload = {}
        try:
            with torch.no_grad():
                for example_index, example in enumerate(self.examples):
                    tensors = generate_cfm_media_tensors(
                        self.model,
                        self.diffusion,
                        example["target_points"],
                        example["image_features"],
                        self.device,
                        self.seed + example_index,
                        self.args.media_guidance_param,
                        self.args.media_cfm_sampling_steps,
                        self.args.media_instantaneous_times,
                    )
                    with _use_pydiffvg_device(self.render_device):
                        tiles = self._render_tiles(example, tensors)
                    overview = _make_overview(tiles)
                    media_name = f"{example_index:02d}_{example['name']}"
                    if self.output_to_disk:
                        self._write_to_disk(global_step, media_name, tiles, overview)
                    if self.output_to_wandb:
                        presentation_tiles = _wandb_tiles(tiles)
                        presentation_overview = _make_overview(presentation_tiles)
                        final_comparison_tiles = _final_comparison_tiles(tiles)
                        final_comparison = None
                        if final_comparison_tiles:
                            final_comparison = _make_overview(
                                final_comparison_tiles,
                                columns=3,
                            )
                        self._add_wandb_images(
                            wandb_payload,
                            media_name,
                            presentation_tiles,
                            presentation_overview,
                            final_comparison,
                        )
        finally:
            self.model.train(was_training)

        if wandb_payload:
            import wandb

            wandb.log(wandb_payload, step=global_step)
        print(
            f"step[{global_step}]: generated fixed CFM validation media "
            f"for {len(self.examples)} example(s)",
            flush=True,
        )

    def _render_tiles(self, example, tensors):
        tiles = [
            ("Input photograph", example["input_image"], None, "input.png"),
        ]
        target_image, target_svg = _render_normalized_points(
            tensors["target"], self.args
        )
        tiles.append(("Ground-truth sketch", target_image, target_svg, "ground_truth.svg"))
        noise_image, noise_svg = _render_normalized_points(
            tensors["initial_noise"], self.args
        )
        tiles.append(("Fixed initial noise", noise_image, noise_svg, "initial_noise.svg"))

        for step_count, sample in tensors["final_samples"].items():
            image, svg = _render_normalized_points(sample, self.args)
            tiles.append(
                (
                    f"CFM {step_count}-step final",
                    image,
                    svg,
                    f"cfm_{step_count}_step_final.svg",
                )
            )

        detailed_steps = tensors["detailed_step_count"]
        for item in tensors["intermediates"]:
            step = item["step"]
            for output_type in ("state", "prediction"):
                image, svg = _render_normalized_points(item[output_type], self.args)
                tiles.append(
                    (
                        f"{detailed_steps}-step {output_type}\nstep {step}/{detailed_steps}",
                        image,
                        svg,
                        os.path.join(
                            f"cfm_{detailed_steps}_step_intermediates",
                            f"step_{step:04d}_of_{detailed_steps:04d}_{output_type}.svg",
                        ),
                    )
                )

        for time_value, prediction in tensors["instantaneous_predictions"].items():
            image, svg = _render_normalized_points(prediction, self.args)
            label = _time_label(time_value)
            tiles.append(
                (
                    f"Instantaneous prediction\nt=r={time_value:g}",
                    image,
                    svg,
                    os.path.join("instantaneous", f"t_{label}_prediction.svg"),
                )
            )
        return tiles

    def _write_to_disk(self, global_step, media_name, tiles, overview):
        example_dir = os.path.join(
            self.output_dir,
            f"step_{global_step:09d}",
            media_name,
        )
        os.makedirs(example_dir, exist_ok=True)
        overview.save(os.path.join(example_dir, "overview.png"))
        for _, image, svg, relative_path in tiles:
            destination = os.path.join(example_dir, relative_path)
            destination_directory = os.path.dirname(destination)
            if destination_directory:
                os.makedirs(destination_directory, exist_ok=True)
            if svg is None:
                image.save(destination)
            else:
                with open(destination, "w") as output_file:
                    output_file.write(svg)

    @staticmethod
    def _add_wandb_images(payload, media_name, tiles, overview, final_comparison):
        import wandb

        prefix = f"Media/{media_name}"
        payload[f"{prefix}/Overview"] = wandb.Image(
            overview,
            caption="Fixed CFM validation overview without initial noise",
        )
        if final_comparison is not None:
            payload[f"{prefix}/Final comparison"] = wandb.Image(
                final_comparison,
                caption="Ground truth, CFM 1-step final, and CFM 4-step final",
            )
        for caption, image, _, relative_path in tiles:
            key = os.path.splitext(relative_path)[0].replace(os.sep, "/")
            payload[f"{prefix}/{key}"] = wandb.Image(image, caption=caption)
