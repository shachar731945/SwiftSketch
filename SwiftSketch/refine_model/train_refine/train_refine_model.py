# This code is based on https://github.com/openai/guided-diffusion

import os
import json
import wandb
from torch.utils.data import DataLoader
from utils.fixseed import fixseed
from utils import dist_util
from utils.sketch_utils import  log_grid_images_list
from utils.model_util import create_model
from refine_model.train_refine.training_loop_refine_model import TrainLoop
from refine_model.utils_refine.parser_util import train_args, get_wandb_name
from refine_model.utils_refine.get_data import create_data_set_refine_model, create_data_set_refine_model_test



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


    print("creating model...", flush=True)
    model = create_model(args)
    model.to(dist_util.dev())

    print("start_data_creation", flush=True)
    train_dataset = create_data_set_refine_model(args.train_data_dir, args.target_key_name, args.diffusion_key_name ,args.image_features_type, args.canvas_width,args.canvas_height, dist_util.dev(), args.scaling_factor, args.cat_data_size, args.sort_by,args.use_data_cache, args.cache_path_dir, args.data_name)
    data= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    validation_data = None
    if args.val_data_dir:
        print("start validation data creation", flush=True)
        validation_dataset = create_data_set_refine_model(
            args.val_data_dir,
            args.target_key_name,
            args.diffusion_key_name,
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
            raise ValueError("The validation dataset is empty. Check --val_data_dir, --target_key_name, and --diffusion_key_name.")
        validation_batch_size = args.val_batch_size or args.batch_size
        validation_data = DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            drop_last=False,
        )

    train_sample= None
    if args.train_sample_dir!= "":
        train_sample_dataset, train_sample_images = create_data_set_refine_model_test(args.train_sample_dir , args.target_key_name, args.diffusion_key_name, args.image_features_type, args.canvas_width,args.canvas_height, dist_util.dev(), args.scaling_factor, args.sort_by)
        train_sample= DataLoader(train_sample_dataset, batch_size=len(train_sample_dataset), shuffle=False)
        log_grid_images_list(train_sample_images, " ", args.use_wandb, log_name="train_sample_images")
        
        
    test=None 
    if args.test_dir!= "":
        test_dataset, test_images = create_data_set_refine_model_test(args.test_dir, args.target_key_name, args.diffusion_key_name ,args.image_features_type, args.canvas_width,args.canvas_height, dist_util.dev(), args.scaling_factor, args.sort_by)
        test= DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        log_grid_images_list(test_images, " ", args.use_wandb, log_name="test_images")


    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0), flush=True)
    print("Training...", flush=True)
    loop= TrainLoop(args, model, data, train_sample, test, validation_data)
    loop.run_loop()
    
    if args.use_wandb:
        wandb.finish()
    

if __name__ == "__main__":
    main()
