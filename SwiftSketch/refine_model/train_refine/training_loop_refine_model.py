
import functools
import os
import numpy as np
import blobfile as bf
import torch
from torch.optim import AdamW
from diffusion import logger
from utils import dist_util
from utils.sketch_utils import rander_image_from_points, render_image_from_norm_points, convert_image_to_pil, log_grid_model_sketches
from diffusion.fp16_util import MixedPrecisionTrainer
from tqdm import tqdm
import wandb
import re
from typing import Optional
from os.path import join as pjoin
from diffusion.Loss_computation import Loss



class TrainLoop:
    def __init__(self, args, model, data , train_sample, test):
        self.args = args
        self.model = model
        self.cond_mode = model.cond_mode
        self.data = data
        self.train_sample = train_sample
        self.test = test
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  
        self.fp16_scale_growth = 1e-3  
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        # assert self.num_steps == 10_000
        # assert len(self.num_epochs) == 1000
        # assert self.num_epochs == 11
   

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())
   
        self.use_ddp = False

        self.loss_func = Loss(args)
        

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.args.save_dir) or self.resume_checkpoint
        print("resume_checkpoint", resume_checkpoint)
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print("resume_step" , self.resume_step)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.args.save_dir) or self.resume_checkpoint
        print("main_checkpoint", main_checkpoint)
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)


    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}', flush=True)
            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                target_control_points, target_rendered_images, diffusion_control_points,image_features = batch
                target_rendered_images = target_rendered_images.permute(0 ,3, 1, 2).to(self.device)
                target_control_points = target_control_points.to(self.device)  # Move to device
                diffusion_control_points = diffusion_control_points.to(self.device)  # Move to device
                image_features=image_features.to(self.device) # Move to device


                self.run_step(target_rendered_images,target_control_points, diffusion_control_points, image_features, step=self.step, resume_step=self.resume_step)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step, v))

                        if k in ['step', 'samples']:
                            continue
                        else:
                            if self.args.use_wandb:
                                wandb.log({f'Loss/{k}': v}, step=self.step)
                
                if (self.step % self.args.log_interval == 0) or (self.step==5):
                    self.model.eval()
                    self.evaluate(self.step, self.resume_step)
                    self.model.train()


                if (self.step % self.save_interval == 0) and (self.step!=0):
                    self.save()

                self.step += 1

            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
   

    def evaluate(self, step, resume_step):
        with torch.no_grad(): 
            for data, data_name in zip([self.train_sample ,self.test ],["train_sample", "test"]) :
                if data is None:
                   continue     
                for batch in data: #only one batch
                        target_control_points, target_rendered_images, diffusion_control_points, diffusion_rendered_images,image_features = batch
                        target_rendered_images = target_rendered_images.permute(0 ,3, 1, 2).to(self.device)
                        target_control_points = target_control_points.to(self.device)  # Move to device
                        diffusion_control_points = diffusion_control_points.to(self.device)  # Move to device
                        image_features=image_features.to(self.device) # Move to device
                        batch_size= target_rendered_images.shape[0]
                        indices_np = np.full((batch_size,), 0) #for exp full noise
                        t = torch.from_numpy(indices_np).long().to(dist_util.dev())
                        compute_losses = functools.partial(
                            self.training_losses,
                            self.model,
                            target_control_points, #  [batch_size, nstrokes, ncpoints, nfeats]
                            diffusion_control_points, #  [batch_size, nstrokes, ncpoints, nfeats]
                            target_rendered_images, #  [bs, canvas_height, canvas_width, 3]
                            image_features, 
                            t,  # [bs](int) sampled timesteps
                            step,
                            resume_step,
                            log_results=True, 
                            log_name= data_name,
                            mode="eval",
                            diffusion_rendered_images= diffusion_rendered_images, #  [bs, canvas_height, canvas_width, 3]
                        )
                    
                        losses = compute_losses()
                        if self.args.use_wandb:
                            wandb_dict={}
                            for k in losses.keys():
                                wandb_dict[f"{data_name}_{k}"] = losses[k]
                            wandb.log(wandb_dict, step=step)

            
        


    def run_step(self, target_rendered_images,target_control_points, diffusion_control_points, image_features, step, resume_step):
        self.forward_backward( target_rendered_images,target_control_points,diffusion_control_points,image_features, step, resume_step)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, target_rendered_images,target_control_points,diffusion_control_points, image_features, step, resume_step):
        self.mp_trainer.zero_grad()
        batch_size= target_rendered_images.shape[0]
        indices_np = np.full((batch_size,), 0) #for exp full noise
        t = torch.from_numpy(indices_np).long().to(dist_util.dev())
  
        compute_losses = functools.partial(
            self.training_losses,
            self.model,
            target_control_points, #  [batch_size, nstrokes, ncpoints, nfeats]
            diffusion_control_points, #  [batch_size, nstrokes, ncpoints, nfeats]
            target_rendered_images, #  [bs, canvas_height, canvas_width, 3]
            image_features, 
            t,  # [bs](int) sampled timesteps
            step,
            resume_step
          
        )
       
        losses = compute_losses()
        loss = (losses["loss"]).mean()

        log_loss_dict(
            {k: v  for k, v in losses.items()}
        )
        self.mp_trainer.backward(loss)


    def training_losses(self, model, target_control_points, diffusion_control_points, target_rendered_images, image_features,t,step, resume_step, log_results=False, log_name="", mode= "train", diffusion_rendered_images=None):
        terms = {}
        model_output = model(x=diffusion_control_points, timesteps=t, image_features=image_features) #shape [bs,nstrokes, ncpoints, nfeats]
       
        target = target_control_points 
        assert model_output.shape == target.shape == target_control_points.shape  # [bs,nstrokes, ncpoints, nfeats]
        model_output_points= model_output

        if log_results and diffusion_rendered_images is not None and self.args.use_wandb: #log model prediction 
            target_sketch_list= render_image_from_norm_points(target_control_points, self.args.scaling_factor, self.args.canvas_width)
            diffusion_sketch_list= [convert_image_to_pil(img) for img in diffusion_rendered_images]
            predict_sketch_list= render_image_from_norm_points(model_output_points, self.args.scaling_factor, self.args.canvas_width)
            log_grid_model_sketches( diffusion_sketch_list,predict_sketch_list,target_sketch_list ,step, log_name=log_name)


        if self.args.lpips_weight:
            #convert the normalized points back to the original range for rendering 
            unnormalized_model_output_points = model_output_points / self.args.scaling_factor
            unnormalized_model_output_points = (unnormalized_model_output_points + 1) / 2
            unnormalized_model_output_points = unnormalized_model_output_points * self.args.canvas_width

            output_rendered_images, _= rander_image_from_points(unnormalized_model_output_points,self.args.canvas_width, self.args.canvas_height)
            output_rendered_images = output_rendered_images.permute(0 ,3, 1, 2).to(target_control_points.device)
        else:
            output_rendered_images= torch.tensor([0.0]).to(target_control_points.device)

        #reshape the points for loss
        bs, nstrokes, ncpoints, nfeats = model_output_points.shape
        model_output_points= model_output_points.reshape(bs, nstrokes, ncpoints*nfeats)# [bs,nstrokes, ncpoints*nfeats (8)]
        target= target.reshape(bs, nstrokes, ncpoints*nfeats)# [bs,nstrokes, ncpoints*nfeats (8)]

        terms = self.loss_func(output_rendered_images,target_rendered_images.detach(), model_output_points, target.detach(), mode= mode)
        terms["loss"] = sum(list(terms.values())) 
        return terms
    


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(save_dir) -> Optional[str]:
    print("find_resume_checkpoint")
    '''look for all file in save directory in the pattent of model{number}.pt
        and return the one with the highest step number.
    '''

    matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(save_dir)}
    models = {int(match.group(1)): file for file, match in matches.items() if match}

    return pjoin(save_dir, models[max(models)]) if models else None


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())

