# SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation

<p align="center">
<img src="docs/swift_teaser.png" width="800px"/>
</p>

*SwiftSketch is a diffusion model that generates vector sketches by denoising a Gaussian in stroke coordinate space. It generalizes effectively across diverse classes and takes under a second to produce a single high-quality sketch.*

#### Ellie Arar, Yarden Frenkel, Daniel Cohen-Or Ariel Shamir, Yael Vinker 

> Recent advancements in large vision-language models have enabled highly expressive and diverse vector sketch generation. However, state-of-the-art methods rely on a time-consuming optimization process involving repeated feedback from a pretrained model to determine stroke placement. Consequently, despite producing impressive sketches, these methods are limited in practical applications. In this work, we introduce SwiftSketch, a diffusion model for image-conditioned vector sketch generation that can produce high-quality sketches in less than a second. SwiftSketch operates by progressively denoising stroke control points sampled from a Gaussian distribution. Its transformer-decoder architecture is designed to effectively handle the discrete nature of vector representation and capture the inherent global dependencies between strokes. To train SwiftSketch, we construct a synthetic dataset of image-sketch pairs, addressing the limitations of existing sketch datasets, which are often created by non-artists and lack professional quality. For generating these synthetic sketches, we introduce ControlSketch, a method that enhances SDS-based techniques by incorporating precise spatial control through a depth-aware ControlNet. We demonstrate that SwiftSketch generalizes across diverse concepts, efficiently producing sketches that combine high fidelity with a natural and visually appealing style.


<a href="https://arxiv.org/abs/2502.08642"><img src="https://img.shields.io/badge/arXiv-2502.08642-b31b1b.svg"></a> 
<a href="https://swiftsketch.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

[**Download the ControlSketch dataset**](https://drive.google.com/drive/folders/1L5kubR416QoTD_UAqH2FtSgNL4leUcys)



## 🔥 NEWS
**`2025/09/27`**: The SwiftSketch code is released!

**`2025/04/29`**: The ControlSketch code is released!

**`2025/02/12`**: The ControlSketch dataset is released!

**`2025/02/12`**: Paper is out!


## Installation

1.  Clone the repo:
```bash
git clone https://github.com/swiftsketch/swiftsketch.git
cd swiftsketch
```

2. Create a new environment:
```bash
conda create -n swiftsketch_env python=3.9.19 -y
conda activate swiftsketch_env
```

3. Install diffvg:
Please follow their [installation guide](https://github.com/BachiLi/diffvg?tab=readme-ov-file#install)


4. Install the libraries:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```


## ControlSketch
```bash
cd ControlSketch
```

To sketch your own image using the optimization method, ControlSketch, from ControlSketch run:
```bash
python object_sketching.py --target <file path>
```
The target can be one of the following:
1. An image file.
2. A dictionary created by the make_sdxl_data.py script, which contains the following keys: image, mask, attn_map, and caption.

The final sketch will be saved in the output_sketches folder.
If the input is a dictionary, the sketch will also be added to the dictionary.

Optional arguments:
* ```--num_strokes``` Defines the number of strokes used to create the sketch, which determines the level of abstraction. The default value is set to 32, but for different images, different numbers might produce better results. 
* ```--fix_scale``` If your image is not squared, it might be resized, it is recommended to use this flag with 1 as input to automatically fix the scale without risuzing the image.
* ```--object_name``` the word for extracting the object cross attention map for strokes intialization. If this is not given, it uses clip attention.
* ```--capiton``` can be a more precise caption of the object and its position. If this is not given, a model will be used to generate the caption.
* ```--use_cpu``` If you want to run the code on the cpu (not recommended as it might be very slow).


<br>
<b>For example, below are optional running configurations:</b>
<br>

Sketching the lion with defauls parameters:
```bash
python object_sketching.py --target "./data/lion.png"
```
Sketching the lion with defauls parameters with a given object_name and a caption:
```bash
python object_sketching.py --target "./data/lion.png" --object_name "lion" --capiton "lion standing"
```
Sketching the cat with defauls parameters using a dictionary target:
```bash
python object_sketching.py --target "./data/cat.npz"
```
Sketching the elephent which is not squared using fix scale:
```bash
python object_sketching.py --target "./data/elephent.png" --fix_scale 1 
```

## Data Creation

To generate a data sample using SDXL run:
```bash
python make_sdxl_data.py --obj <object to generate> 
```
Each generated sample is saved as a dictionary with the following keys:

- **`image`**: The generated image.
- **`mask`**: The corresponding mask.
- **`attn_map`**: The attention map.
- **`caption`**: The caption of the generated image.

The data samples will be saved in the SDXL_samples folder.

Optional arguments:

* ```--num_of_samples```  Number of samples to generate for the given object using different seeds. Default is 1.
* ```--save_compressed_dict```  If set to 0, the output dictionary is saved in uncompressed .npy format. By default, data is saved in compressed .npz format.
* ```--output_dir```  Directory to save the output dictionaries. If not specified, the output will be saved in the SDXL_samples folder.

<br>
<b>For example:</b>
<br>

Generating 10 samples of a cat and save them to a specified directory:
```bash
python make_sdxl_data.py --obj "cat" --output_dir "path/to/output/dir" --num_of_samples 10
```

## SwiftSketch
```bash
cd SwiftSketch
```

###  Download the pretrained models

Download the following models, then unzip and place them in `./save/`. 

[sketch-diffusion](https://drive.google.com/uc?export=download&id=19FryO99dCmz-Dw1jzeZITUI0uuksiOA-)

[refinement-network](https://drive.google.com/uc?export=download&id=1OrLzwaJXZ4SlDw3hqn71Yg1L01ytLv2x)

###  SwiftSketch Generation

To sketch your own images using SwiftSketch, from SwiftSketch run:
```bash
python -m generate \
  --model_path "<path/to/sketch-diffusion_model.pt>" \
  --refine_model_path "<path/to/refinement-network.pt>" \
  --input_data "<path/to/input>" \
  --output_dir "<path/to/output>"
```
The input_data can be one of the following:
1. A single image file
2. A folder of images
3. An .npy/.npz dictionary containing the key image
4. A folder of dictionaries

- The final sketch will be saved in the specified output_dir folder. If no directory is provided, it will be saved in the output_sketches folder inside the input data folder.   
- If the input is a dictionary, the sketch will also be added to the dictionary.

Example:  
sketch all images in the `examples/` folder:

```bash
python -m generate \
    --model_path "./save/sketch-diffusion/model000450000.pt" \
    --refine_model_path "./save/refinement-network/model000430000.pt" \
    --input_data "./examples" \
    --output_dir "./output_sketches"
```

Some example outputs can be found in the output_sketches folder.

###  SwiftSketch Training

<b>Image Features:</b>

To prepare the data for training, you first need to create image features and save them into the input dictionaries by running from the SwiftSketch directory:
```bash
python -m utils.get_features --dir_name <path/to/data>
```
- dir_name is the path to a directory containing .npy/.npz dictionaries.
- The image features key will be added to each dictionary.

<b>Sketch Diffusion Model:</b>

To train the sketch diffusion model, from SwiftSketch run:
```bash
python -m train.train_SwiftSketch --save_dir <path/to/save_dir> --train_data_dir <path/to/training_data> 
```
- The model checkpoints and cached data will be saved in the specified save_dir folder.
- The train_data_dir can be one or more paths to data folders

Optional arguments include:
* ```--num_steps``` Number of training steps
* ```--batch_size``` Batch size used during training
* ```--save_interval``` Save checkpoints every N steps
* ```--lr_schedule exponential``` Exponentially decays the learning rate from `--lr` to `--lr_final_ratio` of `--lr` across `--num_steps`. Set `--lr_final_ratio 0.01` (default) for 1% or `0.02` for 2%. It cannot be combined with ```--lr_anneal_steps```; the program exits with an error before training if both are set.
* ```--val_data_dir``` One or more held-out data directories. When provided, validation losses are logged to Weights & Biases as `Validation/loss`, `Validation/L1_points`, and `Validation/LPIPS`.
* ```--val_interval``` Validation frequency in steps. The default (`0`) uses `--log_interval`; ```--val_max_batches 0``` evaluates the full validation set.
* ```--data_name``` Filename for the cached data
* ```--cat_data_size``` Maximum number of files to use per category (input data path)
* ```--target_key_name``` Name of the target SVG key (the ControlSketch sketch) in the input dictionaries. Default: "svg_32s", consistent with ControlSketch generation.

Example:   
The command below trains a model for 50,000 steps on 1,000 samples each of the cat and dog categories from the controlsketch training set. Both the model checkpoints and the cached data will be saved in the save/cat_dog_model folder. The cached data file will be named cat_dog_data:    
```bash
python -m train.train_SwiftSketch \
    --save_dir "./save/cat_dog_model" \
    --num_steps 50000 \
    --data_name "cat_dog_data" \
    --cat_data_size 1000 \
    --batch_size 16 \ 
    --train_data_dir "./controlsketch_data/train/cat" "./controlsketch_data/train/dog"
```

<b>Cumulative Flow Map DDIM mode:</b>

CFM-DDIM is an additive training and generation mode. The existing DDPM mode
remains the default. A new CFM model can be initialized from a compatible DDPM
checkpoint without restoring the DDPM optimizer or training step:

```bash
python -m train.train_SwiftSketch \
    --diffusion_mode cfm_ddim \
    --init_checkpoint "<path/to/ddpm-model.pt>" \
    --save_dir "<path/to/save_dir>" \
    --train_data_dir "<path/to/training_data>" \
    --val_data_dir "<path/to/validation_data>"
```

The pure implementation optimizes the stopped-target CFM-DDIM MSE from the
paper. `--cfm_instantaneous_prob` defaults to `0.5`, and
`--cfm_loss_weight` defaults to `1.0`. In this mode LPIPS and L1 weights are
automatically forced to zero. Use `--init_checkpoint` for model-only
initialization (fresh optimizer and step 0), and `--resume_checkpoint` to
restore an existing model together with its optimizer and step.

CFM always uses normalized continuous times in `[0, 1]`, with `0` clean and
`1` noisy.
`--diffusion_steps` only selects the internal DDIM schedule discretization;
it no longer changes the CFM time scale or the number of generation calls.
CFM checkpoints created before this normalized-time implementation are
intentionally unsupported.

Generate from a CFM checkpoint with a selectable number of deterministic
cumulative DDIM hops:

```bash
python -m generate \
    --model_path "<path/to/cfm-model.pt>" \
    --input_data "<path/to/input>" \
    --cfm_sampling_steps 1
```

The normalized CFM clock uses clean ground truth `x_0` at time zero and the
noisiest state at time one. One-step generation therefore makes one network
call from `(t, r) = (1, 0)` and adds no reverse-process noise.

<b>Refinement Network:</b>

To prepare the data for training the refinement network, you first need to generate diffusion sketches and save them into the input dictionaries by running:
```bash
python -m generate \
  --model_path "<path/to/sketch-diffusion-model.pt>" \
  --use_refine 0 \
  --save_diffusion_sketch_in_dict 1 \
  --input_data "<path/to/input>"
```
- The generated diffusion SVGs will be added to the input data dictionaries.


To train the refinement network , from SwiftSketch run:
```bash
python -m refine_model.train_refine.train_refine_model \
  --save_dir "<path/to/save_dir>" \
  --init_checkpoint "<path/to/pretrained_sketch_diffusion_model.pt>" \
  --train_data_dir "<path/to/training_data>"
```

- The model checkpoints and cached data will be saved in the specified save_dir folder.
- The train_data_dir can be one or more paths to data folders
- The `--init_checkpoint` argument specifies the pretrained Sketch Diffusion checkpoint. Its model weights initialize the refinement network, while refinement training starts at step 0 with a new optimizer.
- To continue a refinement run, use `--resume_checkpoint` with a refinement-network checkpoint. This restores its model weights, optimizer state, and training step. Do not provide `--init_checkpoint` when resuming.

Optional arguments include:
* ```--num_steps``` Number of training steps
* ```--batch_size``` Batch size used during training
* ```--save_interval``` Save checkpoints every N steps
* ```--lr_schedule exponential``` Exponentially decays the learning rate from `--lr` to `--lr_final_ratio` of `--lr` across `--num_steps`. Set `--lr_final_ratio 0.01` (default) for 1% or `0.02` for 2%. It cannot be combined with ```--lr_anneal_steps```; the program exits with an error before training if both are set.
* ```--val_data_dir``` One or more held-out data directories. When provided, validation losses are logged to Weights & Biases as `Validation/loss`, `Validation/L1_points`, and `Validation/LPIPS`.
* ```--val_interval``` Validation frequency in steps. The default (`0`) uses `--log_interval`; ```--val_max_batches 0``` evaluates the full validation set.
* ```--data_name``` Filename for the cached data
* ```--cat_data_size``` Maximum number of files to use per category (input data path)
* ```--target_key_name``` Name of the target SVG key (the ControlSketch sketch) in the input dictionaries. Default: "svg_32s", consistent with ControlSketch generation.
* ```--diffusion_key_name```Name of the diffusion SVG key (the output of the Sketch Diffusion Model) in the input dictionaries. Default: "svg_diffusion", consistent with SwiftSketch generation when saving the intermediate output- the diffusion SVG.

Example:    
The command below trains the refinement model for 10,000 steps on 1,000 samples each of the cat and dog categories from the controlsketch training set. The model is initialized from the checkpoint saved in ./save/sketch-diffusion/model000450000.pt. Both the model checkpoints and the cached data will be saved in the save_refine/cat_dog_model folder. The cached data file will be named cat_dog_data:   
```bash
python -m refine_model.train_refine.train_refine_model \
    --save_dir "./save/cat_dog_refine_model" \
    --init_checkpoint "./save/sketch-diffusion/model000450000.pt" \
    --num_steps 10000 \
    --data_name "cat_dog_data" \
    --cat_data_size 1000 \
    --batch_size 16 \
    --train_data_dir "./controlsketch_data/train/cat" "./controlsketch_data/train/dog"
```

## Citation
If you make use of our work, please cite our paper:

```
@inproceedings{10.1145/3721238.3730612,
author = {Arar, Ellie and Frenkel, Yarden and Cohen-Or, Daniel and Shamir, Ariel and Vinker, Yael},
title = {SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation},
year = {2025},
isbn = {9798400715402},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3721238.3730612},
doi = {10.1145/3721238.3730612},
booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
articleno = {82},
numpages = {12},
keywords = {Sketch Synthesis, Image-to-Vector Generation, Image-based Rendering, Vector Graphics, Diffusion Models, Stroke-based Representation},
series = {SIGGRAPH Conference Papers '25}
}
```
