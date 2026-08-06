import torch
import math
import torch.nn as nn
import copy
from models.siren import Siren
from models.siren import MultiSiren
from utils.transform_utils import embed2affine
import numpy as np
import os 

class SVR(nn.Module):
    def __init__(self, args, n_slices,steps_total):
        super().__init__()
        self.slice_module = SliceModule(args, n_slices)
        self.args = args
        if args["sr_model"] == "siren":
            self.sr_module = Siren(args["sr_in_dim"], args["sr_out_dim"], args["sr_n_units"],
                            args["sr_n_hidden_layers"], f_om=args["sr_omega_0"],
                            h_om=args["sr_omega_0"], outermost_linear=args["sr_last_linear"])
        elif args["sr_model"] == "multi_siren":
            self.sr_module = MultiSiren(args["sr_in_dim"], args["sr_out_dim"], args["sr_n_units"], args["sr_n_hidden_layers"], 
                            f_om=args["sr_omega_0"],h_om=args["sr_omega_0"], outermost_linear=args["sr_last_linear"],activation='sine')
        
        self.psf_scheduler = PSFScheduler(args, args["epochs"])

    def forward(self, coords,mod_col, affine_nrmd=None, psf_stds=None, phys_bbox=None):
        coords_xyz = self.apply_psf(coords[..., :3], psf_stds, self.psf_scheduler.cardinality) if self.psf_scheduler.cardinality > 0 else coords[...,None, :3]
        coords_xyz_tf, mc, ss, sw = self.slice_module(coords[..., 3:],mod_col,coords_xyz, affine_nrmd, phys_bbox)
        output = self.sr_module(coords_xyz_tf) # output shape: (N, card, sr_out_dim)
        output = output.mean(dim=1) # (N, sr_out_dim)
        return output, mc, ss, sw
    
    def inference(self, coords, psf_stds, psf_card):
        coords = self.apply_psf(coords, psf_stds, psf_card) if psf_card > 0 else coords.unsqueeze(1)
        output = self.sr_module(coords) # output shape: (N, card, sr_out_dim)
        output = output.mean(dim=1)# (N, sr_out_dim)
        return output

    def apply_psf(self, coords, psf_stds, psf_card):
        '''
        Apply PSF to coordinates.
        Args:
            coords: (N, 3) tensor containing coordinates
            psf_stds: (N, 3) tensor containing standard deviations of the PSF
            psf_card: int containing the cardinality of the PSF
        Returns:
            coords_out: (N, card, 3) tensor containing coordinates after PSF application
        '''
        coords_out = coords.unsqueeze(1).expand(-1, psf_card, -1)
        offset = torch.randn(coords_out.shape, dtype=coords_out.dtype, device=coords_out.device) * psf_stds[:, None]
        coords_out = coords_out + offset
        return coords_out
    
    def save_SM(self,path):
        torch.save(self.slice_module.state_dict(), path)


class SliceModule(nn.Module):
    '''
    Slice module for SVR.
    Takes slice indices as input, predicts motion correction (mc) parameters,
    and handles outlier correction via slice weights (`sw`) and slice scalings (`ss`).
    '''
    def __init__(self, args, n_slices):
        super().__init__()
        self.args = args
        self.n_slices = n_slices
        #list of all the slice index
        self.slice_idcs = (torch.arange(n_slices, device=args['device']).unsqueeze(1) / n_slices) * 2 - 1
        self.sm_net = Siren(
            args["sm_in_dim"], args["sm_out_dim"], args["sm_n_units"], args["sm_n_hidden_layers"], 
            args["sm_omega_0"], args["sm_last_linear"], args["sm_omega_schedule"]
        )
    
    def forward(self, coord_slice_idx,mod_col,coords_xyz, affine_normalized=None, phys_bbox=None):
        '''
        Args:
            coord_slice_idx: (N, 1) tensor containing slice indices
            modality_col: (N) tensor containing modality for each slice 
            coords_xyz: (N, 3) tensor containing coordinates
            affine_normalized: (N, 4, 4) tensor containing normalized affine matrix
            phys_bbox: (2, 3) tensor containing physical bounding box of the global space
        Returns:
            coords_out: (N, 3) tensor - coordinates after motion correction
            mc: (N, 6) tensor - motion correction parameters ([:3] = rotation, [3:] = translation)
            sw: (N, 1) tensor - slice weights (sigmoid for [0,1] range)
            ss: (N, 1) tensor - slice scalings (softplus for positive scaling)
        '''
        # Predict motion correction (mc), slice scaling (ss), and slice weighting (sw)
        mc, ss, sw = torch.split(self.sm_net(self.slice_idcs), [6, 1, 1], dim=-1)
        # Apply activation functions
        ss = torch.nn.functional.softplus(ss)  # Ensure positive scaling
        sw = torch.nn.functional.softplus(sw)  # Ensure weight is in [0, 1]

        ##separating with different modalities 
        unique_mods = torch.unique(mod_col) # Get unique modality categories
        mean_values = {mod.item(): ss[mod_col == mod].mean() for mod in unique_mods}  # Compute means
        ## Normalize only within each group TODO: vectorize this
        ss = ss / torch.tensor([mean_values[mod_col[i].item()] for i in range(ss.shape[0])], device=ss.device, dtype=ss.dtype).view(-1, 1) 

        ##For sw
        mean_values = {mod.item(): sw[mod_col == mod].mean() for mod in unique_mods}  # Compute means
        ## Normalize only within each group
        sw = sw / torch.tensor([mean_values[mod_col[i].item()] for i in range(sw.shape[0])], device=sw.device, dtype=sw.dtype).view(-1, 1) 

           
        coord_slice_idx = coord_slice_idx.long().squeeze(-1)
        mc = mc[coord_slice_idx]
        ss = ss[coord_slice_idx]
        sw = sw[coord_slice_idx]
        
        # Convert motion correction to affine transformation
        R, t = embed2affine(mc)  # R is (N, 3, 3), t is (N, 3)
        R = R.unsqueeze(1).expand(-1, coords_xyz.shape[1], -1, -1)
        t = t.unsqueeze(1).expand(-1, coords_xyz.shape[1], -1)
        coords_xyz_tf = torch.einsum('bpxy,bpy->bpx', R, coords_xyz) + t
        if affine_normalized is not None:
            coords_xyz_tf = torch.einsum('bxy,bpy->bpx', affine_normalized[:, :3, :3], coords_xyz_tf) + affine_normalized[:, None, :3, 3]
        if phys_bbox is not None:
            coords_xyz_tf = (coords_xyz_tf - phys_bbox[0]) / (phys_bbox[1] - phys_bbox[0]) * 2 - 1
        return coords_xyz_tf, mc, ss, sw
    
    def save_ss_sw(self,mod_col):
        #save the ss and sw values for all slices at the end of the training
        self.sm_net.eval()
        with torch.no_grad():
            mc, ss, sw = torch.split(self.sm_net(self.slice_idcs), [6, 1, 1], dim=-1)
            ss = torch.nn.functional.softplus(ss)  # Ensure positive scaling
            sw = torch.nn.functional.softplus(sw)  # Ensure weight is in [0, 1]
            sw_og = sw / sw.mean() 
            ss_og = ss / ss.mean()

            #separating with different modalities 
            unique_mods = torch.unique(mod_col) # Get unique modality categories
            mean_values = {mod.item(): ss[mod_col == mod].mean() for mod in unique_mods}  # Compute means
            # Normalize only within each group
            ss = ss / torch.tensor([mean_values[mod_col[i].item()] for i in range(ss.shape[0])], device=ss.device, dtype=ss.dtype).view(-1, 1) 

            #For sw
            mean_values = {mod.item(): sw[mod_col == mod].mean() for mod in unique_mods}  # Compute means
            sw = sw / torch.tensor([mean_values[mod_col[i].item()] for i in range(sw.shape[0])], device=sw.device, dtype=sw.dtype).view(-1, 1) 

            ss_cpu= ss.detach().cpu().numpy()
            sw_cpu= sw.detach().cpu().numpy()
            mc_cpu= mc.detach().cpu().numpy()
            np.save(os.path.join(self.args['data']['recon_dir'], f'ss_mod.npy'), ss_cpu)
            np.save(os.path.join(self.args['data']['recon_dir'], f'sw_mod.npy'), sw_cpu)
            np.save(os.path.join(self.args['data']['recon_dir'], f'mc.npy'), mc_cpu)
            ss_og_cpu= ss_og.detach().cpu().numpy()
            sw_og_cpu= sw_og.detach().cpu().numpy()
            np.save(os.path.join(self.args['data']['recon_dir'], f'ss.npy'), ss_og_cpu)
            np.save(os.path.join(self.args['data']['recon_dir'], f'sw.npy'), sw_og_cpu)



class PSFScheduler(nn.Module):
    def __init__(self, args, steps_total):
        """
        PSF Scheduler for dynamically controlling PSF cardinality during training.

        Args:
            args (dict): Training configuration.
            steps_total (int): Total number of training steps (epochs * steps per epoch).
        """
        super().__init__()
        self.args = args
        self.steps_total = steps_total  # Total number of steps during training
        self.card_start = args["psf_card_start"]
        self.card_end = args["psf_card_end"]
        self.card_diff = self.card_end - self.card_start
        self.card_schedule = args["psf_card_schedule"]
        self.cardinality = self.card_start
        self.steps_current = 0  # Tracks current step

    def update_psf_card(self):
        """Updates the PSF cardinality based on the selected schedule."""
        progress = self.steps_current / self.steps_total  # Normalized progress (0 to 1)

        if self.card_schedule == "constant":
            self.cardinality = self.card_start  # No change in cardinality
        elif self.card_schedule == "linear":
            self.cardinality = self.card_start + self.card_diff * progress
        elif self.card_schedule == "quadratic":
            progress = progress ** 2
            self.cardinality = self.card_start + self.card_diff * progress
        elif self.card_schedule == "cubic":
            progress = progress ** 3
            self.cardinality = self.card_start + self.card_diff * progress
        elif self.card_schedule == "exponential": # exponentially increasing cardinality from card_start to card_end
            base = 1e5  # choose any float > 1; bigger = later + steeper
            progress = (base ** progress - 1) / (base - 1)
            self.cardinality = self.card_start + self.card_diff * progress
        # Ensure cardinality is within range
        self.cardinality = int(min(max(self.cardinality, self.card_start), self.card_end))

        # Increment step counter
        self.steps_current += 1
        return progress

    def forward(self):
        """Returns the updated PSF cardinality for the current training step."""
        self.update_psf_card()
        return int(self.cardinality)  # Ensure integer output
