# This code is based on https://github.com/openai/guided-diffusion

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from refine_model.utils_refine.parser_util import generate_args as refine_generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip, create_model 
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
import utils.sketch_utils as sketch_utils
import model.image_features_models as models
from PIL import Image
import wandb
from transformers import AutoModelForImageSegmentation
import pydiffvg


def main():
    args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)


    if torch.cuda.is_available():
        print("CUDA is available!")

    else:
        print("CUDA is not available.")

    data_path = args.input_data

    if os.path.isfile(data_path):
        # Single file case
        data_dir = os.path.dirname(data_path) or "."  # parent directory
        all_files = [os.path.basename(data_path)]     # list with just this file
    else:
        # Directory case
        data_dir = data_path
        all_files = os.listdir(data_dir)

    print(f"Total all files: {len(all_files)}")

    max_files = 2000  # Maximum number of files to process
    batch_size = args.generate_batch_size   # Number of files per batch
    counter = 0  # To count the processed files
    batch_count=1


    output_path= args.output_dir #path to save the final SVG
    if not output_path:  # empty string or None
        output_path = os.path.join(data_dir, "output_sketches")
    os.makedirs(output_path, exist_ok=True)

    wandb_name = "generate_" + os.path.splitext(os.path.basename(data_dir))[0]
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                    config=args, name=wandb_name, id=wandb.util.generate_id())

    mask_model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True).to(args.device)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    #refine model
    if args.use_refine:
        refine_args= refine_generate_args()
        print("Creating refine model ")
        refine_model = create_model(refine_args)

        print(f"Loading checkpoints (refine model) from [{refine_args.refine_model_path}]...")
        refine_state_dict = torch.load(refine_args.refine_model_path, map_location='cpu')
        load_model_wo_clip(refine_model, refine_state_dict)
        refine_model.to(dist_util.dev())
        refine_model.eval()  


    features_model = models.CLIPMidlleFeutures(args.device, 3).to(args.device) #image features layer 4 


    # Process the files in batches
    for i in range(0, len(all_files), batch_size):
        if counter >= max_files:
            break  # Stop when max files are processed

        # Take a batch of batch_size files
        images_files = all_files[i:i + batch_size]

        # Ensure not to exceed max files
        if counter + len(images_files) > max_files:
            images_files = images_files[:max_files - counter]

        # Update the counter
        counter += len(images_files)

        # Process the current batch
        print(f"Processing batch: {images_files}")

        # Assert that the directory is not empty
        assert len(images_files) > 0, "The test data directory is empty."
 
      
        target_is_dict= os.path.splitext(all_files[0])[-1] == ".npy" or os.path.splitext(all_files[0])[-1] == ".npz" #check if files are dicts or images


        image_features_lst=[]

        if target_is_dict:
            for f_ in os.listdir(data_dir):
                if f_ in images_files:
                    try:
                        read_dictionary= sketch_utils.load_entry(f"{data_dir}/{f_}", [], "CLIPMiddle_layer4_features")
                        if "CLIPMiddle_layer4_features" in read_dictionary.keys():
                            image_features=read_dictionary["CLIPMiddle_layer4_features"].to(args.device)
                        else:
                            input_image = read_dictionary["image"]
                            if "mask" in read_dictionary.keys():
                                mask = read_dictionary["mask"]
                            else:
                                mask= sketch_utils.get_mask(input_image, args.device, mask_model)
                            input_image= sketch_utils.create_masked_image(input_image, mask)
                            if args.fix_scale:
                                input_image=sketch_utils.fix_image_scale(input_image)    
                            image_features = features_model(input_image).to(args.device)
                        image_features_lst.append(image_features)
                    except Exception as e:
                        print(f"Error loading or processing file {f_}: {e}")
                        images_files.remove(f_)


        else:  #if images_files is file of images and not dict:
            for f_ in os.listdir(data_dir):
                if f_ in images_files:
                    try:
                        # Load the image using PIL
                        image_path = os.path.join(data_dir, f_)
                        input_image = Image.open(image_path)
                        print(f"Loaded image: {image_path}")
                        input_image = input_image.convert("RGB")
                        # Force image to match the 1024x1024 mask shape
                        input_image = input_image.resize((1024, 1024))
                        mask= sketch_utils.get_mask(input_image, args.device, mask_model)
                        input_image= sketch_utils.create_masked_image(input_image, mask)
                        if args.fix_scale:
                            input_image=sketch_utils.fix_image_scale(input_image)    
                        image_features = features_model(input_image).to(args.device)
                        image_features_lst.append(image_features)
                    except Exception as e:
                        print(f"Error loading or processing image {f_}: {e}")
                        images_files.remove(f_)
                            

        if len(image_features)==0:
            batch_count+=1
            continue    

        # Assert that number of features is equal to the number of the input images
        assert len(images_files) ==len(image_features_lst), "Error loading or processing the data."

        image_features= torch.stack(image_features_lst)
        image_features= image_features.to(args.device)
        final_batch_size = len(image_features)  
            
        # add CFG scale to batch
        scale= None 
        if args.guidance_param != 1:
            scale = torch.ones(final_batch_size, device=dist_util.dev()) * args.guidance_param
        
        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (final_batch_size, args.num_paths, model.ncpoints, model.nfeats),  
            noise=None,
            clip_denoised=False,
            image_features= image_features,
            scale= scale,
            progress=True,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            dump_steps=None,  
            const_noise=False,
        )

        diffusion_control_points= sample.clone()
              
        if target_is_dict and args.save_diffusion_sketch_in_dict:
            # Save diffusion SVG sketches in dicts
            sample= sketch_utils.denormalize_points(sample, args.scaling_factor, args.canvas_width) #convert the normalized points back to the original range [224,224] 
            _, svg_content_list = sketch_utils.rander_image_from_points(sample,args.canvas_width, args.canvas_height, return_svg_content=True)
            key = f'svg_diffusion'
            for image_file, svg_content in zip(images_files, svg_content_list):
                target_file = f"{data_dir}/{image_file}"
                sketch_utils.save_key(target_file,  svg_content, key)
                print(f"The diffusion SVG was saved to the input dictionary, key is '{key}'")


        if args.use_refine:
            assert len(diffusion_control_points) ==len(image_features), "Error loading or processing the data - refine model."
            batch_size= diffusion_control_points.shape[0]
            indices_np = np.full((batch_size,), 0) #t=0
            t = torch.from_numpy(indices_np).long().to(dist_util.dev())
            refine_model_output = refine_model(x=diffusion_control_points, timesteps=t, image_features=image_features) #shape [bs,nstrokes, ncpoints, nfeats]
            refine_model_output_points= refine_model_output
            refine_model_output_points= sketch_utils.denormalize_points(refine_model_output_points, args.scaling_factor, args.canvas_width) #convert the normalized points back to the original range [224,224] 
            _, final_svg_content_list = sketch_utils.rander_image_from_points(refine_model_output_points,args.canvas_width, args.canvas_height, return_svg_content=True)

            if target_is_dict and args.save_final_sketch_in_dict:
                # Save final SVG sketches in dicts
                key = f'svg_swiftsketch'
                for image_file, svg_content in zip(images_files, final_svg_content_list):
                    target_file = f"{data_dir}/{image_file}"
                    sketch_utils.save_key(target_file, svg_content, key)
                    print(f"The final SwiftSketch SVG was saved to the input dictionary {image_file}, key is '{key}'")

           
            if args.save_svg:
            # Save each SVG content with its corresponding name
                for svg_content, name in zip(final_svg_content_list, images_files):
                    base_name = os.path.splitext(name)[0] 
                    output_file_path = os.path.join(output_path, f"{base_name}.svg")  # Construct file path
                    with open(output_file_path, 'w') as svg_file:
                        svg_file.write(svg_content)  # Write the SVG content to the file

                print(f"SVG files saved in: {output_path}")
        
        print("finish save batch number", batch_count)
        batch_count+=1
    print("finish all")

    if args.use_wandb:
        wandb.finish()



if __name__ == "__main__":
    # pydiffvg.set_use_gpu(torch.cuda.is_available() and use_gpu)
    pydiffvg.set_use_gpu(False)
    # pydiffvg.set_device(args.device)
    pydiffvg.set_device(torch.device("cpu"))
    main()


