# Train for overfit on one example

# Train the actual diffusion model
# python -m train.train_SwiftSketch --save_dir "./train_results/" --num_steps 6000 --data_name "angel_data" --cat_data_size 1 --batch_size 1 --train_data_dir "../dataset_controlsketch/overfit_train_angle/" --use_wandb 1 --wandb_user "shahar_avni-wis" --wandb_project_name "swiftsketch_train"  --log_interval 400 --title "swiftsketch_overfit_train"

# Generate the non refined results
# python -m generate --model_path "./train_results/swiftsketch_overfit_trainCLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000014003.pt" --use_refine 0 --save_diffusion_sketch_in_dict 1 --input_data "../dataset_controlsketch/overfit_train_angle/"

# Train the refinement results
# python -m refine_model.train_refine.train_refine_model \
#     --save_dir "./train_results" \
#     --init_checkpoint "./train_results/swiftsketch_overfit_trainCLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000014003.pt" \
#     --use_data_cache 0 \
#     --num_steps 6000 \
#     --data_name "angel_data" \
#     --cat_data_size 1 \
#     --batch_size 1 \
#     --train_data_dir "../dataset_controlsketch/overfit_train_angle/" \
#     --use_wandb 1 --wandb_user "shahar_avni-wis" --wandb_project_name "swiftsketch_train"  --log_interval 400 --title "swiftsketch_overfit_train"


# Generate using the trained model with overfit to just a single image
# python -m generate \
#     --model_path "./overfit_trained_models/sketch-diffusion/model000014003.pt" \
#     --refine_model_path "./overfit_trained_models/refinement-network/model000020004.pt" \
#     --input_data "../example_run_inputs" \
#     --output_dir "./output_sketches"

#########################################################################################################################

# One class train - currently with val

# Train the actual diffusion model 
# python -m train.train_SwiftSketch --save_dir "./train_results/" --num_steps 7500 --data_name "angel_whole_class_data" --cat_data_size 1000 --batch_size 16 --train_data_dir "../dataset_controlsketch/train/angel" --use_wandb 1 --wandb_user "shahar_avni-wis" --wandb_project_name "swiftsketch_train"  --log_interval 750 --title "swiftsketch_angel_train_with_val" \
#     --val_data_dir  "../dataset_controlsketch/validation/angel" --use_data_cache 0 --save_interval 750

# Trying to continue train with higher learning rate

# python -m train.train_SwiftSketch \
#   --save_dir "./train_results/" \
#   --title "swiftsketch_angel_retrain_from_72802" \
#   --resume_checkpoint "./train_results/swiftsketch_angel_retrain_from_65300CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000072802.pt" \
#   --num_steps 15000 \
#   --data_name "angel_whole_class_data" \
#   --cat_data_size 1000 \
#   --batch_size 16 \
#   --lr 5e-05 \
#   --weight_decay 1e-04 \
#   --train_data_dir "../dataset_controlsketch/train/angel" \
#   --val_data_dir "../dataset_controlsketch/validation/angel" \
#   --use_data_cache 0 \
#   --save_interval 750 \
#   --log_interval 750 \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train"

# Generate the non refined results

#Train
# python -m generate \
#        --model_path "./train_results/swiftsketch_angel_retrain_from_72802CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000073552.pt" \
#        --use_refine 0 --save_diffusion_sketch_in_dict 1 --save_svg 0 --save_final_sketch_in_dict 0 --input_data "../dataset_controlsketch/train/angel/"

# Validation
# python -m generate \
#        --model_path "./train_results/swiftsketch_angel_retrain_from_72802CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000073552.pt" \
#        --use_refine 0 --save_diffusion_sketch_in_dict 1 --save_svg 0 --save_final_sketch_in_dict 0 --input_data "../dataset_controlsketch/validation/angel/"


# Train the refinement results - first run
# python -m refine_model.train_refine.train_refine_model \
#     --save_dir "./train_results" \
#     --init_checkpoint "./train_results/swiftsketch_angel_retrain_from_72802CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000073552.pt" \
#     --use_data_cache 0 \
#     --num_steps 15000 \
#     --data_name "angel_whole_class_data" \
#     --cat_data_size 1000 \
#     --batch_size 16 \
#     --save_interval 750 \
#     --lr_schedule exponential --lr_final_ratio 0.01 --lr 5e-4 --weight_decay 2e-5 \
#     --train_data_dir "../dataset_controlsketch/train/angel/" \
#     --use_wandb 1 --wandb_user "shahar_avni-wis" --wandb_project_name "swiftsketch_train"  --log_interval 750 --title "swiftsketch_angel_train_with_val_refine_from_73552" \
#     --val_data_dir "../dataset_controlsketch/validation/angel" --val_interval 750 \

# Train refinement model better this time - second run
# python -m refine_model.train_refine.train_refine_model \
#     --save_dir "./train_results" \
#     --init_checkpoint "./train_results/swiftsketch_angel_retrain_from_72802CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000073552.pt" \
#     --use_data_cache 0 \
#     --data_name "angel_whole_class_data" \
#     --title "swiftsketch_angel_refine_constant_lr5e6" \
#     --train_data_dir "../dataset_controlsketch/train/angel/" --val_data_dir "../dataset_controlsketch/validation/angel" \
#     --use_wandb 1 --wandb_user "shahar_avni-wis" --wandb_project_name "swiftsketch_train" \
#     --cat_data_size 1000 \
#     --num_steps 30000 \
#     --lr 5e-6 \
#     --weight_decay 2e-6 \
#     --batch_size 16 \
#     --val_interval 500 --save_interval 500 --log_interval 500 \

# Continue after crash
# python -m refine_model.train_refine.train_refine_model \
#   --save_dir "./train_results" \
#   --title "swiftsketch_angel_refine_finetune_from_8000_lr1e6" \
#   --init_checkpoint "./train_results/swiftsketch_angel_refine_constant_lr5e6CLIPMiddle_layer4_seed20_0.2lpips/model000008000.pt" \
#   --use_data_cache 0 \
#   --num_steps 8000 \
#   --data_name "angel_whole_class_data" \
#   --cat_data_size 1000 \
#   --batch_size 16 \
#   --lr 1e-6 \
#   --weight_decay 2e-6 \
#   --save_interval 500 \
#   --log_interval 500 \
#   --val_interval 500 \
#   --train_data_dir "../dataset_controlsketch/train/angel/" \
#   --val_data_dir "../dataset_controlsketch/validation/angel" \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train"

# Train - third try
# python -m refine_model.train_refine.train_refine_model \
#   --save_dir "./train_results" \
#   --title "swiftsketch_angel_refine_lr5e6_wd_2e6" \
#   --init_checkpoint "./train_results/swiftsketch_angel_retrain_from_72802CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000073552.pt" \
#   --use_data_cache 1 \
#   --num_steps 10000 \
#   --data_name "angel_whole_class_data" \
#   --cat_data_size 1000 \
#   --batch_size 16 \
#   --lr 5e-6 \
#   --weight_decay 2e-6 \
#   --save_interval 500 \
#   --log_interval 500 \
#   --val_interval 500 \
#   --train_data_dir "../dataset_controlsketch/train/angel/" \
#   --val_data_dir "../dataset_controlsketch/validation/angel" \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 1

# Train - fourth try
# python -m refine_model.train_refine.train_refine_model \
#   --save_dir "./train_results" \
#   --title "swiftsketch_angel_refine_lr5e6_wd_1e4" \
#   --init_checkpoint "./train_results/swiftsketch_angel_retrain_from_72802CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000073552.pt" \
#   --use_data_cache 1 \
#   --num_steps 10000 \
#   --data_name "angel_whole_class_data" \
#   --cat_data_size 1000 \
#   --batch_size 16 \
#   --lr 5e-6 \
#   --weight_decay 1e-4 \
#   --save_interval 500 \
#   --log_interval 500 \
#   --val_interval 500 \
#   --train_data_dir "../dataset_controlsketch/train/angel/" \
#   --val_data_dir "../dataset_controlsketch/validation/angel" \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 0

# Train - fifth try
# python -m refine_model.train_refine.train_refine_model \
#   --save_dir "./train_results" \
#   --title "swiftsketch_angel_refine_lr6e6_wd_6e5" \
#   --init_checkpoint "./train_results/swiftsketch_angel_retrain_from_72802CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000073552.pt" \
#   --use_data_cache 1 \
#   --num_steps 10000 \
#   --data_name "angel_whole_class_data" \
#   --cat_data_size 1000 \
#   --batch_size 16 \
#   --lr 6e-6 \
#   --weight_decay 6e-5 \
#   --save_interval 500 \
#   --log_interval 500 \
#   --val_interval 500 \
#   --train_data_dir "../dataset_controlsketch/train/angel/" \
#   --val_data_dir "../dataset_controlsketch/validation/angel" \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 1


#########################################################################################################################


# Validation generation of images using the trained model
# python -m generate  --model_path "./angel_with_val_trained_models/sketch-diffusion/model000073552.pt" \
#                     --refine_model_path "./angel_with_val_trained_models/refinement-network/model000008000.pt" \
#                     --use_refine 1 \
#                     --input_data "../dataset_controlsketch/validation/angel" \
#                     --output_dir "./new_output_sketches_drawn_using_model/yes_refine_5.5_guidance" \
#                     --guidance_param 5.5 \
#                     --save_final_sketch_in_dict 0

# python -m generate  --refine_model_path "./angel_trained_models_fron_article/refinement-network/model000430000.pt" \
#                     --model_path "./angel_trained_models_fron_article/sketch-diffusion/model000450000.pt" \
#                     --use_refine 0 \
#                     --input_data "../dataset_controlsketch/validation/angel" \
#                     --output_dir "./angel_output_sketches_original_model_no_refine_guide_2.5" \
#                     --guidance_param 2.5 \
#                     --save_final_sketch_in_dict 0

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
# Now training with cat class instead of angel

# get utils
# python -m utils.get_features \
#   --dir_name ../dataset_controlsketch/train/cat \
#   --network_name CLIPMiddle_layer4

# python -m utils.get_features \
#   --dir_name ../dataset_controlsketch/validation/cat \
#   --network_name CLIPMiddle_layer4

# One class train - currently with val

# Another training configuration

# python -m train.train_SwiftSketch \
#   --save_dir "./train_results/" \
#   --resume_checkpoint "./train_results/ddpm_cat/swiftsketch_cat_lr5e6_wd2e6CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000087564.pt" \
#   --title "swiftsketch_cat_lr5e6_wd2e6" \
#   --num_steps 24000 \
#   --data_name "cat_whole_class_data" --use_data_cache 1 \
#   --cat_data_size 1000 \
#   --batch_size 16 \
#   --lr 5e-06 \
#   --weight_decay 2e-6 \
#   --train_data_dir "../dataset_controlsketch/train/cat" \
#   --val_data_dir "../dataset_controlsketch/validation/cat" \
#   --save_interval 500 \
#   --log_interval 500 \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 0

# python -m train.train_SwiftSketch \
#   --save_dir "./train_results/ddpm_cat" \
#   --title "swiftsketch_cat_lr5e6_wd8e5" \
#   --resume_checkpoint "./train_results/ddpm_cat/swiftsketch_cat_lr5e6_wd8e5CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000100064.pt" \
#   --num_steps 20000 \
#   --data_name "cat_whole_class_data" --use_data_cache 0 \
#   --cat_data_size 1000 \
#   --batch_size 16 \
#   --lr 5e-06 \
#   --weight_decay 8e-5 \
#   --train_data_dir "../dataset_controlsketch/train/cat" \
#   --val_data_dir "../dataset_controlsketch/validation/cat" \
#   --save_interval 500 \
#   --log_interval 500 \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 1

  # python -m train.train_SwiftSketch \
  # --save_dir "./train_results/ddpm_cat" \
  # --title "swiftsketch_cat_lr5e6_wd8e5_continuation_lr7e5_wd8e5_continuation_lr8e5_wd8e5" \
  # --resume_checkpoint "./train_results/ddpm_cat/swiftsketch_cat_lr5e6_wd8e5_continuation_lr7e5_wd8e5_continuation_lr8e5_wd8e5CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000150572.pt" \
  # --num_steps 30000 \
  # --data_name "cat_whole_class_data" --use_data_cache 0 \
  # --cat_data_size 1000 \
  # --batch_size 16 \
  # --lr 8e-05 \
  # --weight_decay 8e-5 \
  # --train_data_dir "../dataset_controlsketch/train/cat" \
  # --val_data_dir "../dataset_controlsketch/validation/cat" \
  # --save_interval 500 \
  # --log_interval 500 \
  # --use_wandb 1 \
  # --wandb_user "shahar_avni-wis" \
  # --wandb_project_name "swiftsketch_train" \
  # --device 1

# Generate the non refined results

# Train
# python -m generate \
#        --model_path "./train_results/ddpm_cat/swiftsketch_cat_lr5e6_wd8e5_continuation_lr7e5_wd8e5_continuation_lr8e5_wd8e5CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000150572.pt" \
#        --use_refine 0 --save_diffusion_sketch_in_dict 1 --save_svg 0 --save_final_sketch_in_dict 0 --input_data "../dataset_controlsketch/train/cat/"

# # Validation
# python -m generate \
#        --model_path "./train_results/ddpm_cat/swiftsketch_cat_lr5e6_wd8e5_continuation_lr7e5_wd8e5_continuation_lr8e5_wd8e5CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000150572.pt" \
#        --use_refine 0 --save_diffusion_sketch_in_dict 1 --save_svg 0 --save_final_sketch_in_dict 0 --input_data "../dataset_controlsketch/validation/cat/"


# Train the refinement results - first run
# python -m refine_model.train_refine.train_refine_model \
#     --save_dir "./train_results/ddpm_cat" \
#     --resume_checkpoint "./train_results/ddpm_cat/cat_refine_5e6_lr_8e5_wdCLIPMiddle_layer4_seed20_0.2lpips/model00002500.pt" \
#     --use_data_cache 0 \
#     --num_steps 80000 \
#     --data_name "cat_whole_class_data" \
#     --cat_data_size 1000 \
#     --batch_size 16 \
#     --lr 5e-06 --weight_decay 8e-5 \
#     --train_data_dir "../dataset_controlsketch/train/cat/" --val_data_dir "../dataset_controlsketch/validation/cat"\
#     --use_wandb 1 --wandb_user "shahar_avni-wis" --wandb_project_name "swiftsketch_train"  --title "cat_refine_5e6_lr_8e5_wd" \
#     --val_interval 500 --log_interval 500 --save_interval 500 \
#     --device 0

# python -m refine_model.train_refine.train_refine_model \
#     --save_dir "./train_results/ddpm_cat" \
#     --init_checkpoint "./train_results/ddpm_cat/swiftsketch_cat_lr5e6_wd8e5_continuation_lr7e5_wd8e5_continuation_lr8e5_wd8e5CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P/model000150572.pt" \
#     --use_data_cache 1 \
#     --num_steps 80000 \
#     --data_name "cat_whole_class_data" \
#     --cat_data_size 1000 \
#     --batch_size 16 \
#     --lr 6.5e-6 --weight_decay 8e-5 \
#     --train_data_dir "../dataset_controlsketch/train/cat/" --val_data_dir "../dataset_controlsketch/validation/cat"\
#     --use_wandb 1 --wandb_user "shahar_avni-wis" --wandb_project_name "swiftsketch_train"  --title "cat_refine_6.5e6_lr_8e5_wd" \
#     --val_interval 500 --log_interval 500 --save_interval 500 \
#     --device 1


#########################################################################################################################


# Validation generation of images using the trained model

# for x in 0 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5; do
#     python -m generate  --model_path "../models/cat_trained_models/sketch-diffusion/model000150572.pt" \
#                         --refine_model_path "../models/cat_trained_models/refinement-network/model000006500.pt" \
#                         --use_refine 1 \
#                         --input_data "../dataset_controlsketch/validation/cat" \
#                         --output_dir "../output_sketches/new_cat/yes_refine_guide_${x}" \
#                         --guidance_param ${x} \
#                         --save_final_sketch_in_dict 0

#     python -m generate  --model_path "../models/cat_trained_models/sketch-diffusion/model000150572.pt" \
#                         --use_refine 0 \
#                         --input_data "../dataset_controlsketch/validation/cat" \
#                         --output_dir "../output_sketches/new_cat/no_refine_guide_${x}" \
#                         --guidance_param ${x} \
#                         --save_final_sketch_in_dict 0
# done

# python -m generate  --refine_model_path "./all_classes_trained_models_fron_article/refinement-network/model000430000.pt" \
#                     --model_path "./all_classes_trained_models_fron_article/sketch-diffusion/model000450000.pt" \
#                     --use_refine 1 \
#                     --input_data "../dataset_controlsketch/validation/cat" \
#                     --output_dir "../output_sketches/cat/article_models_yes_refine_guide_2.5" \
#                     --guidance_param 2.5 \
#                     --save_final_sketch_in_dict 0

# python -m generate  --refine_model_path "./all_classes_trained_models_fron_article/refinement-network/model000430000.pt" \
#                     --model_path "./all_classes_trained_models_fron_article/sketch-diffusion/model000450000.pt" \
#                     --use_refine 0 \
#                     --input_data "../dataset_controlsketch/validation/cat" \
#                     --output_dir "../output_sketches/cat/article_models_no_refine_guide_2.5" \
#                     --guidance_param 2.5 \
#                     --save_final_sketch_in_dict 0


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
# Training angel overfit with the cfm pipeline

# python -m utils.get_features \
#   --dir_name ../dataset_controlsketch/overfit_train_angle \
#   --network_name CLIPMiddle_layer4

# source ../initialize.sh
# python -m train.train_SwiftSketch \
#   --save_dir "./train_results/" \
#   --title "swiftsketch_cfm_angel_overfit_new3" \
#   --num_steps 100000 \
#   --data_name "cfm_angel_overfit_diffusion_data" --use_data_cache 0 \
#   --cat_data_size 1 \
#   --batch_size 1 \
#   --lr 5e-06 \
#   --train_data_dir "../dataset_controlsketch/overfit_train_angel" \
#   --val_data_dir "../dataset_controlsketch/overfit_train_angel" \
#   --save_interval 4000 \
#   --log_interval 4000 \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 0 \
#   --diffusion_mode cfm_ddim \
#   --diffusion_steps 10000 \
#   --cfm_time_samples_per_example 256 \
#   --cfm_val_time_samples_per_example 256 \
#   --cfm_val_instantaneous_prob 0.5

# Try number 2
# source ../initialize.sh
# python -m train.train_SwiftSketch \
#   --save_dir "./train_results/" \
#   --title "swiftsketch_cfm_angel_overfit_new4" \
#   --num_steps 50000 \
#   --data_name "cfm_angel_overfit_diffusion_data" --use_data_cache 0 \
#   --cat_data_size 1 \
#   --batch_size 1 \
#   --lr 2e-06 \
#   --train_data_dir "../dataset_controlsketch/overfit_train_angel" \
#   --val_data_dir "../dataset_controlsketch/overfit_train_angel" \
#   --save_interval 500 \
#   --log_interval 500 \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 1 \
#   --diffusion_mode cfm_ddim \
#   --diffusion_steps 10000 \
#   --cfm_time_samples_per_example 256 \
#   --cfm_val_time_samples_per_example 256 \
#   --cfm_val_instantaneous_prob 0.5

# for cfm_steps in 1 2 3 4 10 50 200 1000; do
#     python -m generate \
#         --model_path "./train_results/swiftsketch_cfm_angel_overfit_new3CLIPMiddle_layer4_seed20_/model000076000.pt" \
#         --use_refine 0 \
#         --input_data "../dataset_controlsketch/overfit_train_angel" \
#         --output_dir "../output_sketches/cfm_angel_overfit_new_${cfm_steps}_steps" \
#         --guidance_param 2.5 \
#         --save_final_sketch_in_dict 0 \
#         --cfm_sampling_steps "${cfm_steps}"
# done

# python -m generate \
#         --model_path "./train_results/swiftsketch_cfm_angel_overfit_new3CLIPMiddle_layer4_seed20_/model000076000.pt" \
#         --use_refine 0 \
#         --input_data "../dataset_controlsketch/overfit_train_angel" \
#         --output_dir "../output_sketches/cfm_new_angel_overfit/4_steps_with_intermediate" \
#         --guidance_param 2.5 \
#         --save_final_sketch_in_dict 0 \
#         --cfm_sampling_steps 4 --save_intermediate_steps 4 --intermediate_output_type "both"

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
# Training angel class with the cfm pipeline

# source ../initialize.sh
python -m train.train_SwiftSketch \
  --save_dir "./train_results/cfm_angel" \
  --title "cfm_angel_from_ddpm_1e6lr_lpips2_article_model_start" \
  --init_checkpoint "../models/angel_with_val_trained_models/sketch-diffusion/model000073552.pt" \
  --num_steps 500000 \
  --data_name "cfm_angel_angel_data" --use_data_cache 1 \
  --cat_data_size 1000 \
  --batch_size 16 --cfm_time_samples_per_example 1 --cfm_instantaneous_prob 0.5 \
  --val_batch_size 4 --cfm_val_time_samples_per_example 4 --cfm_val_instantaneous_prob 0.5 \
  --lr 1e-06 \
  --train_data_dir "../dataset_controlsketch/train/angel" \
  --val_data_dir "../dataset_controlsketch/validation/angel" \
  --save_interval 2000 \
  --log_interval 500 \
  --use_wandb 1 \
  --wandb_user "shahar_avni-wis" \
  --wandb_project_name "swiftsketch_train" \
  --device 0 \
  --diffusion_mode cfm_ddim \
  --diffusion_steps 50 \
  --normalize_model_output 1 \
  --lpips_weight 1 --l1_points_weight 0 \
  --media_interval 5000 \
  --media_output_mode both \
  --media_cfm_sampling_steps 1 4 \
  --media_instantaneous_times 0.25 0.5 0.75 1.0 \
  --media_guidance_param 1 \
  --media_log_at_start 1

# source ../initialize.sh
# python -m train.train_SwiftSketch \
#   --save_dir "./train_results/cfm_angel" \
#   --resume_checkpoint "./train_results/cfm_angel/cfm_angel_class_1e5_lrCLIPMiddle_layer4_seed20_/model000025000.pt" \
#   --title "cfm_angel_class_1e5_lr" \
#   --num_steps 500000 \
#   --data_name "cfm_angel_angel_data" --use_data_cache 1 \
#   --cat_data_size 1000 \
#   --batch_size 10 \
#   --lr 1e-05 \
#   --train_data_dir "../dataset_controlsketch/train/angel" \
#   --val_data_dir "../dataset_controlsketch/validation/angel" \
#   --save_interval 2000 \
#   --log_interval 1000 \
#   --use_wandb 1 \
#   --wandb_user "shahar_avni-wis" \
#   --wandb_project_name "swiftsketch_train" \
#   --device 1 \
#   --diffusion_mode cfm_ddim \
#   --diffusion_steps 10000 \
#   --cfm_time_samples_per_example 80 \
#   --cfm_val_time_samples_per_example 80 \
#   --cfm_val_instantaneous_prob 0.5

# for cfm_steps in 1 4 20; do
#   for guidance_param in 2.5; do
#     python -m generate \
#         --model_path "./train_results/cfm_angel/cfm_angel_class_5e6_lrCLIPMiddle_layer4_seed20_/model000035000.pt" \
#         --use_refine 0 \
#         --input_data "../dataset_controlsketch/train/angel" \
#         --output_dir "../output_sketches/cfm_angel/train_5e6_lr/cfm_angel_${cfm_steps}_steps_${guidance_param}_gp" \
#         --guidance_param "${guidance_param}" \
#         --save_final_sketch_in_dict 0 \
#         --cfm_sampling_steps "${cfm_steps}"

#     python -m generate \
#         --model_path "./train_results/cfm_angel/cfm_angel_class_5e6_lrCLIPMiddle_layer4_seed20_/model000035000.pt" \
#         --use_refine 0 \
#         --input_data "../dataset_controlsketch/validation/angel" \
#         --output_dir "../output_sketches/cfm_angel/val_5e6_lr/cfm_angel_${cfm_steps}_steps_${guidance_param}_gp" \
#         --guidance_param "${guidance_param}" \
#         --save_final_sketch_in_dict 0 \
#         --cfm_sampling_steps "${cfm_steps}"
#   done
# done
