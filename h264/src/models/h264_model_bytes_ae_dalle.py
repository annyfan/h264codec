import os
from contextlib import nullcontext, contextmanager

import fsspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from h264.src.models.utils import exists, restore_parts
from h264.src.models.modules.byteformer.byteformer import ByteFormer
from h264.src.models.modules.deconv.aedalledecoder import DeconvDecoder, ReduceEmbeding
from h264.src.models.utils import split_batch, unnormalize_zero_to_one
from h264.src.scheduler.cosine import CosineScheduler
import pytorch_warmup as warmup
from glob import glob
from h264.src.scheduler.fixed import FixedLRScheduler
from h264.src.scheduler.reducelronplateau import ReduceLROnPlateau
from h264.src.utils.common import get_value_from_config

from h264.src.models.utils import normalize_neg_one_to_one
from h264.src.models.modules.losses.msssim_l1 import MS_SSIM_L1_LOSS
from h264.src.models.modules.losses.vggloss import VGGLoss
from h264.src.models.modules.losses.msssim import MSSSIM
class BytesDeconv(nn.Module):
    """
    Model class ...

    """

    def __init__(self, config: dict) -> None:
        super(BytesDeconv, self).__init__()
        self.config = config
        self.device = config['model']['device']
        self.byteformer = ByteFormer(config)
        self.reduction_net  = ReduceEmbeding(hidden_dims =config["model"]["decovdecoder"]["reduction_dim"],
                                            dropout = config["model"]["decovdecoder"].get("dropout", 0.0) )
        self.decoder = DeconvDecoder(config)


    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """ return (recon_x, input, mu, log_var, z)"""

        output = self.byteformer(input_data['h264'])
        output_reduced = self.reduction_net(output['last_hidden_state'], output['attn_mask'])
       
               
        recon_x =  self.decoder(output_reduced, )
        
        return recon_x
    


class ModelTrainer(nn.Module):
    """
    Returned elements for the model building.
    """

    def __init__(self, config, model: BytesDeconv, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.config = config
        self.model = model
        self.can_checkpoint = get_value_from_config(config, 'train.can_checkpoint', False)
        self.fs = fsspec.filesystem('file')
        lr = get_value_from_config(config, 'train.learning_rate', 1e-4)
        eps = get_value_from_config(config, 'train.eps', 1e-8)
        beta1 = get_value_from_config(config, 'train.beta1', 0.9)
        beta2 = get_value_from_config(config, 'train.beta2', 0.99)
        max_grad_norm = get_value_from_config(config, 'train.max_grad_norm', None)
        warmup_steps = get_value_from_config(config, 'train.warmup_steps', None)
        cosine_decay_max_steps = get_value_from_config(config, 'train.cosine_decay_max_steps', None)


        warmup_steps = get_value_from_config(config, 'train.scheduler.warmup_steps', 0)
        cosine_decay_max_steps = get_value_from_config(config, 'train.scheduler.cosine_decay_max_steps', 150000)
        warmup_init_lr = get_value_from_config(config, 'train.scheduler.warmup_init_lr', 1e-7)
        cosine_min_lr=get_value_from_config(config, 'train.scheduler.cosine_min_lr', 1e-5)
        cosine_max_lr=get_value_from_config(config, 'train.scheduler.cosine_max_lr', 0.4)
        self.scheduler_name = get_value_from_config(config, 'train.scheduler.name', 'cosine')


        verbose = get_value_from_config(config, 'train.verbose', True)
        optimizer_name = get_value_from_config(config, 'train.optimizer', 'adam')

        self.device = get_value_from_config(config, 'model.device', "cpu")


        
        self.grad_scaler_enabled = True

        if optimizer_name == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                eps=eps,
                betas=(beta1, beta2),
                weight_decay=0.01
            )
        else :
            self.optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                eps=eps,
                betas=(beta1, beta2)
            )

        # gradient clipping if needed
        self.max_grad_norm = max_grad_norm

        self.scaler = GradScaler(enabled=self.grad_scaler_enabled)

        self.scheduler = self.warmup_scheduler = None

        if  self.scheduler_name == 'cosine':
            self.scheduler = CosineScheduler(warmup_iterations = warmup_steps, warmup_init_lr = warmup_init_lr,
                                              max_iterations = cosine_decay_max_steps, min_lr=cosine_min_lr, max_lr=cosine_max_lr)
        elif self.scheduler_name == "fix":
            self.scheduler = FixedLRScheduler(warmup_iterations = warmup_steps, warmup_init_lr = warmup_init_lr,
                                                        max_iterations = config["train"]["scheduler"]["max_steps"], lr=config["train"]["scheduler"]["lr"])
        elif self.scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer= self.optimizer,
                warmup_iterations = warmup_steps, 
                warmup_init_lr = warmup_init_lr,
                max_epochs = config["train"]["scheduler"]["max_epochs"],
                lr_min=config["train"]["scheduler"]["lr_min"], 
                lr_max=lr,
                factor=config["train"]["scheduler"]["factor"],
                patience=config["train"]["scheduler"]["patience"], 
                dont_halve_until_epoch=config["train"]["scheduler"]["dont_halve_until_epoch"],
            )
        else:
                           
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_decay_max_steps)
            if exists(warmup_steps):
                self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=warmup_steps)
          
        
        self.verbose = verbose
        self.model.to(self.device)
        self.to(self.device)
        self.max_checkpoints_keep = 1

        self.reconstruction_loss =  config["model"]["decovdecoder"]["reconstruction_loss"] 
        if self.reconstruction_loss == "mssim":
            self.mssim_loss = MSSSIM(config["model"]["decovdecoder"].get("window_size", 11),
                                config["model"]["decovdecoder"].get("size_average", True), 2)
        elif self.reconstruction_loss == "mssiml1":
            self.mssiml1_loss = MS_SSIM_L1_LOSS(
                config["model"]["decovdecoder"].get("gaussian_sigmas", [0.5, 1.0, 2.0, 4.0, 8.0]),
                data_range=2,
                K=(0.01, 0.03),
                alpha = config["model"]["decovdecoder"].get("alpha", 0.025),
                compensation=200.0, device=config["model"]["device"])
        
        self.feature_loss = config["model"]["decovdecoder"].get("feature_loss", None)
        if  config["model"]["decovdecoder"].get("feature_loss", None) == "vgg":
            self.crit_vgg = VGGLoss(layer=config["model"]["decovdecoder"].get("vgg_layer", 'relu2_2'), device=config["model"]["device"])
            self.vgg_weight = config["model"]["decovdecoder"].get("vgg_weight", 1.0)
            #crit_tv = vgg_loss.TVLoss(p=1)
     

    def update(self,  epoch_idx, curr_iter):


        # set the grad scaler on the accelerator, since we are managing one per u-net

        if exists(self.max_grad_norm):
            # For gradient clipping, unscale the gradients and then clip them
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )

        self.optimizer.step()
        self.optimizer.zero_grad()

        # scheduler, if needed
        if  self.scheduler_name in  ["cosine", "fix"] :
            self.optimizer = self.scheduler.update_lr(
                optimizer=self.optimizer, epoch=epoch_idx, curr_iter=curr_iter
            )
        if  self.scheduler_name in  ["plateau"] :
            self.optimizer = self.scheduler.update_lr( optimizer=self.optimizer, epoch=epoch_idx, curr_iter=curr_iter, current_loss=float("-inf"))
        #else:
        #    maybe_warmup_context = nullcontext() if not exists(self.warmup_scheduler) else self.warmup_scheduler.dampening()
        #    with maybe_warmup_context:
        #        if exists(self.scheduler):
        #            self.scheduler.step()


    def get_lr(self):
        if exists(self.optimizer):
            return self.optimizer.param_groups[0]['lr']
        return 0.0

    
    def forward(
            self,
            batch=None
    ):
  
        max_batch_size = get_value_from_config(self.config, 'train.max_batch_size', None)

        total_loss = 0.
        total_loss_recons = 0.

        for chunk_size_frac, chunked_batch in split_batch(split_size=max_batch_size, batch=batch):
            recon_x = self.model(input_data=chunked_batch)
            losses =  self.loss_function(recon_x, chunked_batch['image'])

            loss = losses['loss'] * chunk_size_frac
            loss_recons = losses["mse_loss"] * chunk_size_frac
            total_loss += loss.item()
            total_loss_recons += loss_recons.item()

            if self.training:
                self.scaler.scale(loss).backward()

        return total_loss , total_loss_recons

    def loss_function(self, recon_x, image):
        image_neg_one_to_one =  normalize_neg_one_to_one(image)
        mse_loss = F.mse_loss(
                recon_x.reshape(image_neg_one_to_one.shape[0], -1),
                image_neg_one_to_one.reshape(image_neg_one_to_one.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        mse_loss = mse_loss.mean(dim=0)
        
        if self.reconstruction_loss == "mse":
            recon_loss = mse_loss
        elif self.reconstruction_loss == "mssim":
            recon_loss = self.mssim_loss(recon_x, image_neg_one_to_one)
        elif self.reconstruction_loss == "mssiml1":
            recon_loss = self.mssiml1_loss(recon_x, image_neg_one_to_one)


        loss = recon_loss

        if self.feature_loss == "vgg":
            vgg_loss = self.crit_vgg(unnormalize_zero_to_one(recon_x), image)
            loss += vgg_loss * self.vgg_weight

        return { 
            'recon_loss':recon_loss,
            'loss':loss,
            'recon_x':recon_x,
            'mse_loss' : mse_loss
        }
    
    def save(
            self,
            path,
            overwrite=True,
            without_optim_and_sched=False,
            **kwargs
    ):

        fs = self.fs
        save_obj = dict(
            model=self.model.state_dict(),
            version=0.1,
            optimizer = self.optimizer.state_dict(),
            **kwargs
        )

        with fs.open(path, 'wb') as f:
            torch.save(save_obj, f)

        print(f'checkpoint saved to {path}')

    def load(self, path, only_model=False, strict=True, noop_if_not_exist=False):
        fs = self.fs

        if noop_if_not_exist and not fs.exists(path):
            print(f'trainer checkpoint not found at {str(path)}')
            return

        assert fs.exists(path), f'{path} does not exist'


        with fs.open(path) as f:
            loaded_obj = torch.load(f, map_location='cpu')

        try:
            self.model.load_state_dict(loaded_obj['model'], strict=strict)
            if 'optimizer' in loaded_obj:
                self.optimizer.load_state_dict(loaded_obj['optimizer'])
        except RuntimeError as e:
            print("Failed loading state dict. Trying partial load")
            self.model.load_state_dict(restore_parts(self.model.state_dict(),
                                                     loaded_obj['model']))
            #raise e

        if only_model:
            return loaded_obj

        print(f'checkpoint loaded from {path}')
        return loaded_obj

    def save_to_checkpoint_folder(self, checkpoint_path):
        if not self.can_checkpoint:
            return

        self.save(checkpoint_path)

        if self.max_checkpoints_keep <= 0:
            return

        sorted_checkpoints = self.all_checkpoints_sorted(checkpoint_path)
        if len(sorted_checkpoints) > self.max_checkpoints_keep:
            checkpoints_to_discard = sorted_checkpoints[self.max_checkpoints_keep:]
            for checkpoint in checkpoints_to_discard:
                self.fs.rm(checkpoint)

    def all_checkpoints_sorted(self, checkpoint_path):
        glob_pattern = os.path.join(checkpoint_path, 'checkpoint.*.pt')
        checkpoints = glob(glob_pattern)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: int(str(x).split('.')[-2]), reverse=True)
        return sorted_checkpoints

    def load_from_checkpoint_folder(self, checkpoint_path, last_total_steps=-1, ):
        if last_total_steps != -1:
            filepath = os.path.join(checkpoint_path, f'checkpoint.{last_total_steps}.pt')
            self.load(filepath)
            return

        sorted_checkpoints = self.all_checkpoints_sorted(checkpoint_path)

        if len(sorted_checkpoints) == 0:
            print(f'no checkpoints found to load from at {checkpoint_path}')
            return

        last_checkpoint = sorted_checkpoints[0]
        self.load(last_checkpoint)

    @torch.no_grad()
    def sample(self, x:  torch.Tensor, retloss = False, **kwargs)-> torch.Tensor:
        """
        Args:
            x : The training input

        Returns:
            recon_x

        """

        recon_x =  self.model(x)
        recon_x =  unnormalize_zero_to_one(recon_x)

        mse_loss = 0.0
        if retloss:
            image = x["image"].to(self.config["model"]["device"], non_blocking=True)
            image_neg_one_to_one =  normalize_neg_one_to_one(image)
            recon_x_neg_one_to_one = normalize_neg_one_to_one(recon_x)
            mse_loss = F.mse_loss(
                recon_x_neg_one_to_one.reshape(image.shape[0], -1),
                image_neg_one_to_one.reshape(image.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        return {'recon_x': recon_x, 'losses': mse_loss}
    

def compute_loss(
        model_trainer: ModelTrainer, data: dict, config: dict
) -> torch.Tensor:
    """ """
    image = data["image"].to(config["model"]["device"], non_blocking=True)
    h264 = data["h264"].to(config["model"]["device"], non_blocking=True)

    loss, recons_loss = model_trainer.forward(batch={"image": image, "h264": h264})

    return loss, recons_loss


# Create H264 model
def create_model(config: dict) -> ModelTrainer:
    """
    Inception model builder.

    Args:
        config: the global parameters for the experiment.

    Returns:
        Returns the information related to the PyTorch model, scheduler, optimizer, and scaler.
    """
    
    model = BytesDeconv(config)
    model = torch.nn.DataParallel(model)
    model.to(config["model"]["device"])
    # Return the model, the optimizer, the scheduler and scaler.
    model_trainer = ModelTrainer(config, model)
    return model_trainer
