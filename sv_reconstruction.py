import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from utils.loss_functions import LossComposition
from dataprocessing.data_processor import DataProcessor
from models.inr_svr import SVR
from utils import utils
import nibabel as nib
import os
import math
import matplotlib.pyplot as plt
import numpy as np

class SVR_Trainer:
    def __init__(self, args):
        self.args = args
        self._init_data()
        self._init_model()
        self._init_optimizer()
        self.loss_comp = LossComposition(args)
        self.scaler = GradScaler()  # Initialize gradient scaler for mixed precision

    def _init_data(self):
        self.dataprocessor = DataProcessor(self.args).to(self.args['device'])

    def _init_model(self):
        steps_total = self.args['epochs'] * self.dataprocessor.steps_per_epoch # total number of steps
        if self.args['load_model']:
            self.svr_model = torch.load(self.args['model_path']).to(self.args['device'])
        else:
            self.svr_model = SVR(self.args, self.dataprocessor.n_slices_global, steps_total).to(self.args['device'])
        
        self.svr_model = torch.compile(self.svr_model, disable=not self.args['compile'])

    def _init_optimizer(self):
        self.param_groups = [
            {'params': self.svr_model.sr_module.parameters(), 'lr': self.args['lr_sr']},
            {'params': self.svr_model.slice_module.parameters(), 'lr': self.args['lr_sm']}
        ]
        self.optimizer = optim.Adam(self.param_groups)

        if self.args['scheduler'] == 'none':
            self.scheduler = None
        elif self.args['scheduler'] == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=self.args['step_size'], gamma=self.args['gamma'])
        elif self.args['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, eta_min=self.args['lr_sr']*0.5, T_max=self.args['epochs'])
        else:
            raise ValueError(f"Scheduler {self.args['scheduler']} not supported")

    def train(self):
        self.svr_model.train()
        progress = 0.0
        phys_bbox = self.dataprocessor.phys_bbox
        epoch_iterator = tqdm(range(self.args['epochs']), desc="Training Progress", position=0, leave=True)
        loss_history = []  # Store loss values for analysis
        loss_history_regul = []  # Store regularization loss values for analysis
        for epoch in epoch_iterator:
            epoch_loss = 0.0
            epoch_loss_regul= 0.0
            # Iterate over dataset
            for batch in self.dataprocessor:
                coords, values, value_mask, affine_nrmd, psf_stds = batch # values is a tensor of shape (N, C) C is the number of channels, e.g. echo time GT values
                progress=1.0 #for adult data
                psf_stds = psf_stds * progress # change adult here 1.0
                self.optimizer.zero_grad()
                # Automatic mixed precision training
                with autocast(enabled=self.args.get('use_amp', True)):
                    values_p, mc, ss, sw = self.svr_model(coords,self.dataprocessor.slices_modalities,affine_nrmd, psf_stds, phys_bbox) 
                    #loss, _ = self.loss_comp(values_p, values, value_mask, mc, ss, sw, epoch, self.args['data']['coeff'],self.args['data']['echotime']) 
                    coeff = self.args['data']['coeff']
                    if coeff != 0:
                        loss,regul,_ = self.loss_comp(values_p, values, value_mask, mc, ss, sw, epoch, coeff,self.args['data']['echotime']) 
                        epoch_loss_regul += regul.item()
                    else:
                        loss, _, _ = self.loss_comp(values_p, values, value_mask, mc, ss, sw, epoch, coeff=0, echotime=self.args['data']['echotime'])
               #if loss nan stop
                # Scaled backpropagation
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item()

            # Step the learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
            progress = self.svr_model.psf_scheduler.update_psf_card() # update the PSF cardinality
             
            # Update tqdm with the latest loss (instead of printing)
            epoch_iterator.set_postfix(loss=f"{epoch_loss / self.dataprocessor.steps_per_epoch:.6f}")
                    # Compute average loss for the epoch
            avg_loss = epoch_loss / self.dataprocessor.steps_per_epoch
            loss_history.append(avg_loss)
            #for only regul
            avg_loss_regul = epoch_loss_regul / self.dataprocessor.steps_per_epoch
            loss_history_regul.append(avg_loss_regul)      
            #      
            # Update tqdm with the latest loss (instead of printing)
            epoch_iterator.set_postfix(loss=f"{epoch_loss / self.dataprocessor.steps_per_epoch:.6f}")
        #print ss et sw for the last epoch for all slices
        self.svr_model.slice_module.save_ss_sw(self.dataprocessor.slices_modalities)

        plt.figure()
        #plt.plot(np.log(np.array(loss_history)))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss + Regul')
        plt.title('Training Loss')
        plt.savefig(self.args['data']['recon_dir'] + '/loss.png')
        np.save(self.args['data']['recon_dir'] + '/loss.npy', np.array(loss_history))
        self.reconstruct(psf_card=0)

    def reconstruct(self, psf_card=None):
        '''
        Reconstruct the image volume from the SVR model by querying the model at a grid of coordinates.
        '''
        bs = self.args['batch_size']
        args_data = self.args['data']
        self.svr_model.eval()
        with torch.no_grad():
            coords, psf_stds, recon_shape, affine = self.dataprocessor.get_full_coordinate_grid(device=self.args['device'])
            values_p = torch.empty((len(coords), self.args['sr_out_dim']), device=self.args['device'])
            for i in range(len(coords) // bs + 1): # iterate over the full grid in batches
                coords_batch = coords[i*bs:(i+1)*bs]
                psf_stds_batch = psf_stds[i*bs:(i+1)*bs]
                values_p_batch = self.svr_model.inference(coords_batch, psf_stds_batch, psf_card=psf_card)
                # assert no nan or inf in values_p_batch
                assert not torch.isnan(values_p_batch).any(), "NaN values in values_p_batch"
                assert not torch.isinf(values_p_batch).any(), "Inf values in values_p_batch"
                values_p[i*bs:(i+1)*bs] = values_p_batch

            values_p = values_p.clamp(args_data['normalize_values'][0], args_data['normalize_values'][1])
            # project to 0, 1
            values_p = (values_p - args_data['normalize_values'][0]) / (args_data['normalize_values'][1] - args_data['normalize_values'][0])
            values_p = values_p.view(recon_shape+(self.args['sr_out_dim'],)).cpu().numpy()

            for i in range(self.args['sr_out_dim']):
                img_nii = nib.Nifti1Image(values_p[..., i], affine.detach().cpu().numpy())
                save_to = os.path.join(args_data['recon_dir'], utils.get_subject_name(args_data['path_stacks'][0]) + f"_recon_{i}.nii.gz")
                nib.save(img_nii, save_to)
                print(f"Reconstructed image saved to {save_to}")
    
    def reconstruct_simulated_stacks(self, save_prefix="simulated_stack", psf_scale=1.0):
        """
        Recreate simulated slices from the trained model and write them back into
        full 3D stack volumes using the ORIGINAL stack affine of each input stack.

        Also saves a second stack containing sw values.
        """
        self.svr_model.eval()
        phys_bbox = self.dataprocessor.phys_bbox
        recon_dir = self.args['data']['recon_dir']

        with torch.no_grad():
            for stack_id, stack in enumerate(self.dataprocessor.stacks):
                stack_shape = stack.stack_img.shape[:3]

                sim_stack = np.zeros(stack_shape, dtype=np.float32)
                sim_sw_stack = np.zeros(stack_shape, dtype=np.float32)

                for slice_obj in stack.slices:
                    global_idx = slice_obj.global_idx
                    meta = self.dataprocessor.get_slice_meta(global_idx)
                    out_channel = meta["out_channel"]
                    non_zero_xy = meta["non_zero_vxls_2d"]

                    n_voxels = meta["end_idx"] - meta["start_idx"]

                    pred_non_zero = torch.empty(
                        (n_voxels,),
                        device=self.args['device'],
                        dtype=torch.float32
                    )

                    sw_non_zero = torch.empty(
                        (n_voxels,),
                        device=self.args['device'],
                        dtype=torch.float32
                    )

                    write_ptr = 0

                    for j in range(0, n_voxels, self.args['batch_size']):
                        s = meta["start_idx"] + j
                        e = min(meta["start_idx"] + j + self.args['batch_size'], meta["end_idx"])

                        coords = self.dataprocessor.coords[s:e]
                        affine_nrmd = self.dataprocessor.affine_nrmd[s:e]
                        psf_stds = self.dataprocessor.psf_stds[s:e] * psf_scale

                        values_p, mc, ss, sw = self.svr_model(
                            coords,
                            self.dataprocessor.slices_modalities,
                            affine_nrmd,
                            psf_stds,
                            phys_bbox
                        )

                        pred_chunk = values_p[:, out_channel]

                        # Adjust this depending on the shape of sw returned by your model
                        if sw.ndim > 1:
                            sw_chunk = sw[:, 0]
                        else:
                            sw_chunk = sw

                        chunk_len = len(pred_chunk)
                        pred_non_zero[write_ptr:write_ptr + chunk_len] = pred_chunk
                        sw_non_zero[write_ptr:write_ptr + chunk_len] = sw_chunk
                        write_ptr += chunk_len

                    sim_slice = np.zeros(meta["img_shape"], dtype=np.float32)
                    sw_slice = np.zeros(meta["img_shape"], dtype=np.float32)

                    pred_non_zero_np = pred_non_zero.detach().cpu().numpy()
                    sw_non_zero_np = sw_non_zero.detach().cpu().numpy()

                    x = non_zero_xy[:, 0]
                    y = non_zero_xy[:, 1]

                    sim_slice[x, y] = pred_non_zero_np
                    sw_slice[x, y] = sw_non_zero_np

                    sim_stack[:, :, meta["local_idx"]] = sim_slice
                    sim_sw_stack[:, :, meta["local_idx"]] = sw_slice

                if self.args['data']['normalize_values']:
                    vmin, vmax = self.args['data']['normalize_values']
                    sim_stack = np.clip(sim_stack, vmin, vmax)

                img_nii = nib.Nifti1Image(sim_stack, stack.stack_img.affine, stack.stack_img.header)
                sw_nii = nib.Nifti1Image(sim_sw_stack, stack.stack_img.affine, stack.stack_img.header)

                img_save_to = os.path.join(recon_dir, f"{save_prefix}_{stack_id}.nii.gz")
                sw_save_to = os.path.join(recon_dir, f"{save_prefix}_sw_{stack_id}.nii.gz")

                nib.save(img_nii, img_save_to)
                nib.save(sw_nii, sw_save_to)

            print(f"Simulated stack saved to {img_save_to}")
            print(f"SW stack saved to {sw_save_to}")

    def save_model(self, path):
        torch.save(self.svr_model.state_dict(), path)
