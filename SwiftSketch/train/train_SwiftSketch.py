# This code is based on https://github.com/openai/guided-diffusion

import os
import json
from torch.utils.data import DataLoader
import wandb

from utils.fixseed import fixseed
from utils.parser_util import train_args, get_wandb_name
from utils import dist_util
from train.training_loop import TrainLoop
from utils.model_util import create_model_and_diffusion
from utils.get_data import create_data_set


def main():
    args = train_args()
    fixseed(args.seed)

    wandb_name= get_wandb_name(args)
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                    config=args, name=wandb_name,id=wandb.util.generate_id()
                    )

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    if not args.cache_path_dir:
        args.cache_path_dir= args.save_dir

    args.save_dir = os.path.join(args.save_dir, wandb_name)


    if os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)


    print("start data creation", flush=True)
    train_dataset = create_data_set(args.train_data_dir, args.target_key_name ,args.image_features_type, args.canvas_width,args.canvas_height, dist_util.dev(), args.scaling_factor, args.cat_data_size ,args.sort_by, args.use_data_cache, args.cache_path_dir, args.data_name)
    data= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    validation_data = None
    if args.val_data_dir:
        print("start validation data creation", flush=True)
        validation_dataset = create_data_set(
            args.val_data_dir,
            args.target_key_name,
            args.image_features_type,
            args.canvas_width,
            args.canvas_height,
            dist_util.dev(),
            args.scaling_factor,
            args.cat_data_size,
            args.sort_by,
            args.use_data_cache,
            args.cache_path_dir,
            f"{args.data_name}_validation",
        )
        if len(validation_dataset) == 0:
            raise ValueError("The validation dataset is empty. Check --val_data_dir and --target_key_name.")
        validation_batch_size = args.val_batch_size or args.batch_size
        validation_data = DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    print("creating model and diffusion...", flush=True)
    model, diffusion = create_model_and_diffusion(args)
    model.to(dist_util.dev())

    

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0), flush=True)
    print("Training...", flush=True)
    loop= TrainLoop(args, model, diffusion, data, validation_data)
    loop.run_loop()
    

    if args.use_wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()
