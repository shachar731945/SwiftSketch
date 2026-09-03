from argparse import ArgumentParser
import argparse
import os
import json
import sys


CFM_TIME_VERSION = 3


def get_cfm_instantaneous_samples_per_example(
    time_samples_per_example,
    instantaneous_probability,
    sample_count_argument="--cfm_time_samples_per_example",
    probability_argument="--cfm_instantaneous_prob",
):
    """Return the exact instantaneous count, or ``None`` for legacy K=1 sampling."""
    if time_samples_per_example < 1:
        raise ValueError(f"{sample_count_argument} must be at least 1.")
    if not 0.0 <= instantaneous_probability <= 1.0:
        raise ValueError(f"{probability_argument} must be between 0 and 1.")

    # K=1 is the backward-compatible mode: there is only one Bernoulli draw,
    # so a non-trivial probability cannot be represented exactly per example.
    if time_samples_per_example == 1:
        return None

    expected_count = time_samples_per_example * instantaneous_probability
    instantaneous_count = round(expected_count)
    if abs(expected_count - instantaneous_count) > 1e-9:
        raise ValueError(
            f"{sample_count_argument}={time_samples_per_example} cannot "
            f"represent {probability_argument}={instantaneous_probability} "
            "exactly. Choose a sample count whose product with the probability "
            "is an integer."
        )
    return instantaneous_count


def resolve_cfm_validation_sampling(
    training_samples_per_example,
    training_instantaneous_probability,
    validation_samples_per_example,
    validation_instantaneous_probability,
):
    """Resolve inherited CFM validation settings and enforce split availability."""
    samples_per_example = (
        training_samples_per_example
        if validation_samples_per_example == 0
        else validation_samples_per_example
    )
    instantaneous_probability = (
        training_instantaneous_probability
        if validation_instantaneous_probability is None
        else validation_instantaneous_probability
    )
    get_cfm_instantaneous_samples_per_example(
        samples_per_example,
        instantaneous_probability,
        sample_count_argument="--cfm_val_time_samples_per_example",
        probability_argument="--cfm_val_instantaneous_prob",
    )
    if samples_per_example == 1 and 0.0 < instantaneous_probability < 1.0:
        raise ValueError(
            "The effective --cfm_val_time_samples_per_example must be at least "
            "2 when --cfm_val_instantaneous_prob is strictly between 0 and 1."
        )
    return samples_per_example, instantaneous_probability



def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sketch_options(parser)
    add_loss_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion', 'sketch', 'loss']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    if (
        model_args.get("diffusion_mode") == "cfm_ddim"
        and model_args.get("cfm_time_version") != CFM_TIME_VERSION
    ):
        raise ValueError(
            "This CFM checkpoint uses an incompatible time-conditioning "
            "implementation and is intentionally unsupported."
        )

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path', required=True)
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')
    
def get_wandb_name(args):
        losses_weights= [args.lpips_weight, args.l1_points_weight]
        losses_name= ["lpips","L1P"]
        lst=[]
        for i in range(len(losses_weights)):
            if losses_weights[i]:
                lst.append(f"{losses_weights[i]}{losses_name[i]}")
        losses_str = "_".join(lst)

        wandb_name = f"{args.title}{args.image_features_type}_seed{args.seed}_"+losses_str
        return wandb_name


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=20, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--diffusion_mode", default="ddpm",
                       choices=["ddpm", "cfm_ddim"], type=str,
                       help="Use the legacy DDPM objective/sampler or the cumulative-flow DDIM objective.")
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--cos_power", default=0.4 , type=float,
                       help="The power of the cosine in the noise_schedule")
    group.add_argument("--diffusion_steps", default=50, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--model_mean_type", default='start_x',choices=['start_x', 'epsilon'], type=str, 
                       help="what is the model prediction.")
    group.add_argument("--cfm_instantaneous_prob", default=0.5, type=float,
                       help="Probability of replacing the CFM endpoint r with t during training.")
    group.add_argument("--cfm_loss_weight", default=1.0, type=float,
                       help="Multiplier for the pure CFM-DDIM MSE objective.")

    


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_dec',
                       choices=['trans_enc', 'trans_dec'], type=str)
    group.add_argument("--emb_trans_dec", default=0, type=int,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--normalize_model_output", default=1, type=int,
                       help="if true, will normalize the output of the model to be in the data scale ")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--heads", default=4, type=int,
                       help="Number of heads.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--image_features_type", default="CLIPMiddle_layer4", type=str,
                       help="Type of image features.")
    
    
def add_sketch_options(parser):
    group = parser.add_argument_group('sketch')
    group.add_argument("--num_paths", type=int,
                        default=32, help="number of strokes")
    group.add_argument("--width", type=float,
                        default=2.5, help="stroke width")
    group.add_argument("--canvas_width", type=int, default=224,
                       help="The width hight for the loss")
    group.add_argument("--canvas_height", type=int, default=224,
                       help="The canvas hight for the loss, needs to be equal to canvas_width")


def add_loss_options(parser):
    group = parser.add_argument_group('loss')
    group.add_argument("--lpips_weight", type=float, default=0.2,
                        help="weight the lpips loss")
    group.add_argument("--l1_points_weight", type=float, default=1.0,
                        help="weight the l1 loss")
   

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    parser.add_argument('--train_data_dir', type=str, nargs='+', default=[],
                        help="List of training data directories")
    parser.add_argument('--val_data_dir', type=str, nargs='+', default=[],
                        help="Optional list of validation data directories. If omitted, validation is disabled.")
    group.add_argument("--cat_data_size", default=10000, type=int,
                       help="The maximum number of files to use per category (input data path)")
    group.add_argument("--scaling_factor", default=2.0, type=float,
                       help="The data will be normalized to [-1,1]*scaling_factor")
    group.add_argument("--use_data_cache", default=1, type=int,
                    help="Set to 1 to use cached data, 0 to disable caching.")
    group.add_argument("--cache_path_dir", default="", type=str,
                   help="Directory path to save the data cache. If not given, uses the save_dir")
    group.add_argument("--data_name", default="data", type=str,
                    help="Filename for the cached data.")
    group.add_argument("--target_key_name", default="svg_32s", type=str,
                    help="The name of the target SVG key in the input dicts.")
    
      

def add_wandb_options(parser):
    group = parser.add_argument_group('wandb')
    group.add_argument("--use_wandb", type=int, default=0)
    group.add_argument("--wandb_user", type=str, default="")
    group.add_argument("--wandb_name", type=str, default="test")
    group.add_argument("--wandb_project_name", type=str, default="SwiftSketch")
    group.add_argument("--experiment_name", type=str, default="SwiftSketch")
    group.add_argument("--title", type=str, default="", help="Prefix for wandb run name and save directory.")



def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save data cache, checkpoints and results.")
    group.add_argument("--overwrite", type=int, default=1,
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=5e-05, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--lr_schedule", default="none", choices=["none", "exponential"], type=str,
                       help="Learning-rate schedule. Exponential decays from --lr to --lr_final_ratio of --lr over --num_steps.")
    group.add_argument("--lr_final_ratio", default=0.01, type=float,
                       help="Final LR divided by initial LR for --lr_schedule exponential (for example, 0.01 or 0.02).")
    group.add_argument("--log_interval", default=5_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--val_interval", default=0, type=int,
                       help="Run validation each N steps. Set to 0 to use --log_interval.")
    group.add_argument("--val_batch_size", default=0, type=int,
                       help="Validation batch size. Set to 0 to use --batch_size.")
    group.add_argument("--val_max_batches", default=0, type=int,
                       help="Maximum validation batches per evaluation. Set to 0 to use the full validation set.")
    group.add_argument("--val_seed", default=1234, type=int,
                       help="Seed for deterministic validation timesteps and diffusion noise.")
    group.add_argument("--cfm_val_time_samples_per_example", default=0, type=int,
                       help="Independent CFM time/noise samples per validation example. "
                            "Set to 0 to inherit --cfm_time_samples_per_example.")
    group.add_argument("--cfm_val_instantaneous_prob", default=None, type=float,
                       help="Validation probability for t=r. Omit to inherit --cfm_instantaneous_prob.")
    group.add_argument("--save_interval", default=10_000, type=int,
                       help="Save checkpoints each N steps")
    group.add_argument("--num_steps", default=60_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--cfm_time_samples_per_example", default=1, type=int,
                       help="Number of independent CFM (t, r, noise) samples evaluated in parallel per data example. "
                            "Values above 1 must exactly represent --cfm_instantaneous_prob.")
    group.add_argument("--media_interval", default=0, type=int,
                       help="Generate CFM validation media each N training steps. 0 disables media generation.")
    group.add_argument("--media_validation_files", default=[], nargs="+", type=str,
                       help="Optional fixed .npy/.npz validation files used for media. "
                            "When omitted, the first valid sorted file under --val_data_dir is used.")
    group.add_argument("--media_seed", default=None, type=int,
                       help="Seed for fixed media noise. Omit to inherit --val_seed.")
    group.add_argument("--media_guidance_param", default=2.5, type=float,
                       help="Classifier-free guidance scale used only for validation media generation.")
    group.add_argument("--media_cfm_sampling_steps", default=[1, 4], nargs="+", type=int,
                       help="CFM sampling step counts shown in validation media.")
    group.add_argument("--media_instantaneous_times", default=[0.25, 0.5, 0.75, 1.0], nargs="+", type=float,
                       help="Times t=r at which to show instantaneous clean predictions.")
    group.add_argument("--media_log_at_start", default=1, choices=[0, 1], type=int,
                       help="Log fixed validation media before the first optimizer update.")
    group.add_argument("--media_output_mode", default="both",
                       choices=["wandb", "disk", "both"], type=str,
                       help="Write validation media to W&B, the run directory, or both.")
    group.add_argument("--media_output_dir", default="", type=str,
                       help="Media output directory. Empty uses <run save_dir>/media.")
    group.add_argument("--validation_media_render_device", default="cpu",
                       choices=["cpu", "model"], type=str,
                       help="Render validation-media sketches on CPU or on the training model's device.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--init_checkpoint", default="", type=str,
                       help="Initialize model weights only from a checkpoint; starts with a fresh optimizer and step 0.")
    group.add_argument("--sort_by", default="no_sorting", type=str, 
                       choices=["highest_point", "length", "contour_and_attn", "no_sorting"],
                       help="sorting criterion for the data. ")
                        # If the data was created using ControlSketch, it is already sorted (contour_and_attn).



def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--input_data", required=True, type=str,
                       help="Path to a dir of images files/ npy dicts/ npz dicts or path to a file of an image/ npy dict/npz dict") #used for generating new sketch for images
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (if not given, a 'results' folder will be created inside the data folder). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--generate_batch_size", default=24, type=int, help="Batch size during generation.")
    group.add_argument("--use_refine", default=1, type=int,
                       help="If 1, use the full SwiftSketch pipeline with the refinement network. If 0, use only the diffusion inference process.")
    group.add_argument("--fix_scale", type=int, default=0, help="if the target image is not squared, it is recommended to fix the scale")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--save_final_sketch_in_dict", default=1, type=int,
                       help="If 1 and the input is a dict, save the final SwiftSketch SVG into the input dict.")
    group.add_argument("--save_svg", default=1, type=int,
                       help="If 1, save the final SwiftSketch SVG results into the output_dir.")
    group.add_argument("--save_diffusion_sketch_in_dict", default=0, type=int,
                       help="If 1 and the input is a dict, save the diffusion process SVG into the input dict.")
    group.add_argument("--refine_model_path", default='',  type=str,
                       help="Path to refine model####.pt file to be sampled.")
    group.add_argument("--cfm_sampling_steps", default=1, type=int,
                       help="Number of deterministic cumulative DDIM hops for a CFM checkpoint.")
    group.add_argument("--save_intermediate_steps", default=0, type=int,
                       help="Save this many evenly spaced sampling positions (including the final position). "
                            "0 keeps final-only generation.")
    group.add_argument("--intermediate_output_type", default="both", type=str,
                       choices=["both", "state", "prediction"],
                       help="At each saved sampling position, write the current state, the predicted clean sketch, or both.")
    
   
 

def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    else:
        cond_mode = 'image'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_sketch_options(parser)
    add_loss_options(parser)
    add_wandb_options(parser)

    args = parser.parse_args()
    if args.media_interval < 0:
        parser.error("--media_interval must be 0 or a positive integer.")
    if args.media_interval:
        if args.diffusion_mode != "cfm_ddim":
            parser.error("Training validation media currently requires --diffusion_mode cfm_ddim.")
        if not args.media_validation_files and not args.val_data_dir:
            parser.error("Training validation media requires --media_validation_files or --val_data_dir.")
        if args.media_output_mode == "wandb" and not args.use_wandb:
            parser.error("--media_output_mode wandb requires --use_wandb 1.")
        if not args.media_cfm_sampling_steps:
            parser.error("--media_cfm_sampling_steps must contain at least one step count.")
        if any(
            step_count < 1 or step_count > args.diffusion_steps
            for step_count in args.media_cfm_sampling_steps
        ):
            parser.error(
                "Every --media_cfm_sampling_steps value must be between 1 and --diffusion_steps."
            )
        if not args.media_instantaneous_times:
            parser.error("--media_instantaneous_times must contain at least one time.")
        if any(time < 0.0 or time > 1.0 for time in args.media_instantaneous_times):
            parser.error("Every --media_instantaneous_times value must be between 0 and 1.")
    if args.diffusion_mode == "cfm_ddim":
        # Persist a checkpoint-format marker. Generation and resume reject
        # earlier CFM checkpoint formats rather than interpreting them with a
        # different time-conditioning convention.
        args.cfm_time_version = CFM_TIME_VERSION
        if args.model_mean_type != "start_x":
            parser.error("--diffusion_mode cfm_ddim currently requires --model_mean_type start_x.")
        if args.diffusion_steps < 2:
            parser.error("--diffusion_mode cfm_ddim requires --diffusion_steps of at least 2.")
        if not 0.0 <= args.cfm_instantaneous_prob <= 1.0:
            parser.error("--cfm_instantaneous_prob must be between 0 and 1.")
        cfm_lpips_enabled = args.lpips_weight > 0
        if args.lpips_weight < 0:
            parser.error("--lpips_weight must be non-negative.")
        if args.l1_points_weight:
            parser.error(
                "CFM-DDIM does not use the L1 point loss; set --l1_points_weight 0."
            )
        if cfm_lpips_enabled:
            if args.cfm_time_samples_per_example != 1:
                parser.error(
                    "CFM-DDIM with LPIPS requires --cfm_time_samples_per_example 1."
                )
            effective_val_samples = (
                args.cfm_val_time_samples_per_example
                or args.cfm_time_samples_per_example
            )
            if effective_val_samples != 1:
                parser.error(
                    "CFM-DDIM with LPIPS requires --cfm_val_time_samples_per_example "
                    "0 or 1; validation evaluates every endpoint regime explicitly."
                )
            if args.normalize_model_output != 1:
                parser.error(
                    "CFM-DDIM with LPIPS requires --normalize_model_output 1 so the "
                    "existing tanh output normalization bounds rendered points."
                )
            validation_probability = (
                args.cfm_instantaneous_prob
                if args.cfm_val_instantaneous_prob is None
                else args.cfm_val_instantaneous_prob
            )
            if not 0.0 <= validation_probability <= 1.0:
                parser.error("--cfm_val_instantaneous_prob must be between 0 and 1.")
        else:
            try:
                get_cfm_instantaneous_samples_per_example(
                    args.cfm_time_samples_per_example,
                    args.cfm_instantaneous_prob,
                )
                if args.val_data_dir:
                    resolve_cfm_validation_sampling(
                        args.cfm_time_samples_per_example,
                        args.cfm_instantaneous_prob,
                        args.cfm_val_time_samples_per_example,
                        args.cfm_val_instantaneous_prob,
                    )
            except ValueError as error:
                parser.error(str(error))
        if args.cfm_loss_weight <= 0:
            parser.error("--cfm_loss_weight must be positive.")
    if args.init_checkpoint and args.resume_checkpoint:
        parser.error("--init_checkpoint and --resume_checkpoint are mutually exclusive.")
    if args.lr_schedule == "exponential":
        lr_anneal_was_specified = any(
            argument == "--lr_anneal_steps" or argument.startswith("--lr_anneal_steps=")
            for argument in sys.argv[1:]
        )
        if lr_anneal_was_specified:
            parser.error("--lr_schedule exponential cannot be combined with --lr_anneal_steps. "
                         "Remove one schedule option before training.")
        if not 0 < args.lr_final_ratio < 1:
            parser.error("--lr_final_ratio must be greater than 0 and less than 1 when --lr_schedule exponential is used.")
    return args


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_wandb_options(parser)
    add_base_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)

    if args.diffusion_mode == "cfm_ddim":
        if args.model_mean_type != "start_x":
            parser.error("CFM-DDIM generation requires a start_x checkpoint.")
        if not 1 <= args.cfm_sampling_steps <= args.diffusion_steps:
            parser.error("--cfm_sampling_steps must be between 1 and --diffusion_steps.")
    if args.save_intermediate_steps < 0:
        parser.error("--save_intermediate_steps must be 0 or a positive integer.")
    sampling_steps = (
        args.cfm_sampling_steps
        if args.diffusion_mode == "cfm_ddim"
        else args.diffusion_steps
    )
    if args.save_intermediate_steps > sampling_steps:
        parser.error(
            "--save_intermediate_steps cannot exceed the number of sampling "
            "steps for the selected diffusion mode."
        )

    return args
