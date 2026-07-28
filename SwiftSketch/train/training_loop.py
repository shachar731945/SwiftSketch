
import functools
import os
import blobfile as bf
import torch
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import re
from typing import Optional
from os.path import join as pjoin

from diffusion.resample import create_named_schedule_sampler
from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler


class TrainLoop:
    def __init__(self, args, model, diffusion, data):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
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
        # print(f"THEHEHEHEHEHE DATATATA SIZEZEZEZEZ ISISISISISISIS {len(self.data)}")
        # assert len(self.data) == 1_000
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

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.use_ddp = False
        

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.args.save_dir) or self.resume_checkpoint
        # print("resume_checkpoint", resume_checkpoint, flush=True)
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print("resume_step" , self.resume_step, flush=True)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.args.save_dir) or self.resume_checkpoint
        print("main_checkpoint", main_checkpoint, flush=True)
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

                target_control_points, target_rendered_images, image_features = batch
                target_rendered_images = target_rendered_images.permute(0 ,3, 1, 2).to(self.device)
                target_control_points = target_control_points.to(self.device)  # Move to device
                image_features=image_features.to(self.device) # Move to device

               
                self.run_step(target_rendered_images,target_control_points, image_features, step=self.step, resume_step=self.resume_step)
                
                  
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v), flush=True)

                        if k in ['step', 'samples']:
                            continue
                        else:
                            if self.args.use_wandb:
                                wandb.log({f'Loss/{k}': v}, step=self.step+self.resume_step)


                if (self.step % self.save_interval == 0) and (self.step!=0):
                    self.save()
             
                self.step += 1

            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, target_rendered_images,target_control_points, image_features, step, resume_step):
        self.forward_backward( target_rendered_images,target_control_points, image_features, step, resume_step)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

       
        

    def forward_backward(self, target_rendered_images,target_control_points, image_features, step, resume_step):
        self.mp_trainer.zero_grad()
        t, weights = self.schedule_sampler.sample(target_rendered_images.shape[0], dist_util.dev())
  
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            target_control_points, #  [batch_size, nstrokes, ncpoints, nfeats]
            target_rendered_images, #  [bs, canvas_height, canvas_width, 3]
            image_features, 
            t,  # [bs](int) sampled timesteps
            step,
            resume_step
          
        )
       

        losses = compute_losses()
        
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
   

        loss = (losses["loss"] * weights).mean()

        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        self.mp_trainer.backward(loss)

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
    print("find_resume_checkpoint", flush=True)
    '''look for all file in save directory in the pattent of model{number}.pt
        and return the one with the highest step number.
    '''

    matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(save_dir)}
    models = {int(match.group(1)): file for file, match in matches.items() if match}

    return pjoin(save_dir, models[max(models)]) if models else None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
