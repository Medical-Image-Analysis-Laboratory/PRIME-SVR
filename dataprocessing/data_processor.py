import nibabel as nib
import numpy as np
import torch
import ants
from torch.utils.data import Dataset, DataLoader
from utils.transform_utils import get_coordinate_grid, get_psf_stds, torch_hom_matrix, torch_inv_hom_matrix
from dataprocessing.data_utils import align_by_center_of_mass
import os 

class Stack:
    '''
        Stack class to store the stack information.
        Contains methods for pre-processing, loading, and getting slice indices.
    '''
    def __init__(self, args, stack_id, slice_id_offset, ref_stack=None):
        '''
            paths: tuple of paths to the stack (img, mask)
            stack_id: id of the stack, used to identify the stack in the DataProcessor
            slice_id_offset: offset of the slice id, i.e., number of slices already seen in previous stacks
            ref_stack: reference stack, used to align the stack to the reference stack
        '''
        # init attributes
        self.args = args
        self.paths = (args['data']['path_stacks'][stack_id], args['data']['path_masks'][stack_id] if args['data']['path_masks'] else None)
        self.stack_id = stack_id
        self.slice_id_offset = slice_id_offset
        self.stack_img, self.stack_mask = self.load_stack(ref_stack)
        self._init_affines()
        self._init_slices()
        self._init_phys_bbox()
        self._init_psf()

    def _init_affines(self):
        self.vxl_shape = np.array(self.stack_img.shape[:3])
        self.spacing = np.array(self.stack_img.header.get_zooms()[:3])
        self.affine = self.stack_img.affine
        self.affine_nrmd = self.affine.copy()
        self.affine_nrmd[:3, :3] = self.affine_nrmd[:3, :3] @ np.diag(1 / self.spacing) # normalize affine to voxel space (1mm spacing)
        phys_center = np.array(self.vxl_shape - 1) / 2
        phys_center = self.affine_nrmd[:3, :3] @ (phys_center * self.spacing)
        self.affine_nrmd[:3, 3] += phys_center # correct center 0 shift in origin (i.e., since origin vxl becomes -1*center voxel, correct world coordinate)

    def _init_phys_bbox(self):
        slice_bbox = np.concatenate([slice.phys_slice_bbox for slice in self.slices], axis=0)
        self.phys_bbox = np.stack([slice_bbox.min(axis=0), slice_bbox.max(axis=0)])

    def load_stack(self, ref_stack=None):
        stack_img = nib.load(self.paths[0])
        if self.args['data']['bias_field_correction'] == 'stack_wise':
            stack_img = ants.n4_bias_field_correction(ants.from_nibabel(stack_img)).to_nibabel()

        stack_img_data = stack_img.get_fdata()
        stack_mask = nib.load(self.paths[1]) if self.paths[1] is not None else nib.Nifti1Image((stack_img_data > 0).astype(np.int32), stack_img.affine)
        stack_img_data = np.clip(stack_img_data, a_min=0, a_max=np.percentile(stack_img_data, 99.9))
        stack_mask_data = (stack_mask.get_fdata()>0).squeeze()

        #if external min max provided
        min_, max_ = stack_img_data[stack_mask_data].min(), stack_img_data[stack_mask_data].max()
        stack_mask = nib.Nifti1Image(stack_mask_data.astype(np.int32), stack_mask.affine)
        stack_img = nib.Nifti1Image(stack_img_data, stack_img.affine)
        if self.args['data']['align_stacks'] == 'center_of_mass' and ref_stack is not None:
            stack_img, stack_mask = align_by_center_of_mass(stack_img, stack_mask, ref_stack.stack_mask, self.stack_id, self.args['data']['recon_dir'])
        return stack_img, stack_mask
    
    def _init_slices(self):
        '''
            Initializes the slices of the stack.
            Only slices with non-zero values are considered.
        '''
        self.slices = [] # list of non-zero slices
        for i in range(self.vxl_shape[2]):
            slice = Slice(stack=self, local_idx=i, global_idx=len(self.slices)+self.slice_id_offset)
            if len(slice.non_zero_vxls) > 0:
                self.slices.append(slice)
        self.n_slices = len(self.slices) # number of non-zero slices

    def _init_psf(self):
        '''
        Initializes the Point Spread Function (PSF) using a Gaussian approximation.

        The PSF is computed based on in-plane voxel resolution and slice thickness
        using the standard convention in the literature.
        '''
        slice_thickness = self.args['data']['slice_thickness'][self.stack_id] if self.args['data']['slice_thickness'] is not None else None
        self.psf_stds = get_psf_stds(self.spacing, slice_thickness)

        
class Slice:
    '''
        Slice class to store the slice information.
    '''
    def __init__(self, stack, local_idx, global_idx):
        '''
            stack: Stack object
            local_idx: id of the slice, used to identify the slice in the Stack
            global_idx: global id of the slice, used to identify the slice in the DataProcessor 
                             across stacks (only slices with non-zero values are considered!)
            img: image of the slice
            mask: mask of the slice
            non_zero_vxls: non-zero voxel coordinates in the slice
            non_zero_values: non-zero voxel values in the slice
            coordinates: coordinates of the non-zero voxels in the slice
        '''
        self.stack = stack
        self.local_idx = local_idx
        self.global_idx = global_idx
        self.img = self.stack.stack_img.get_fdata()[:, :, self.local_idx]
        self.mask = self.stack.stack_mask.get_fdata()[:, :, self.local_idx]
        self.non_zero_vxls, self.non_zero_values = self.get_non_zero_voxels_and_values()
        self.coordinates = self._process_coordinates()
        self._init_slice_bbox()

    def preprocess_slice(self):
        if self.stack.args['data']['bias_field_correction'] == 'slice_wise':
            self.img = ants.n4_bias_field_correction(ants.from_numpy(self.img)).to_numpy()

    def get_non_zero_voxels_and_values(self):
        '''
        Returns:
            - np.ndarray of shape (N, 3), where each row is (x, y, z)
              representing non-zero voxel coordinates in the slice.
        '''
        non_zero_voxels = np.array(np.nonzero(self.mask)).T  # Shape (N, 2)
        non_zero_values = self.img[non_zero_voxels[:, 0], non_zero_voxels[:, 1]]
        # Append slice_id as the third dimension to make (x, y, slice_id)
        slice_column = np.full((non_zero_voxels.shape[0], 1), self.local_idx)  # Shape (N, 1)
        # Concatenate along columns to create (x, y, z)
        return np.hstack((non_zero_voxels, slice_column)), non_zero_values

    def _process_coordinates(self):
        '''
        Transformation of voxel coordinates to world space and normalization.

        Returns:
            torch.Tensor: Transformed voxel coordinates.
        '''
        coordinates = self.non_zero_vxls # shape (N, 3)
        if self.stack.args['data']['use_world_space']:
            coordinates = np.hstack((coordinates, np.ones((self.non_zero_vxls.shape[0], 1)))) # shape (N, 4)
            coordinates = (self.stack.affine @ coordinates.T).T[:, :3]  # Convert to world space
             # **Center coordinates by the stack's world space center**
            stack_center = (self.stack.affine @ np.append(np.array(self.stack.vxl_shape) / 2, 1))[:3]
            coordinates = coordinates - stack_center  # Centering
        else:
            # center coordinates by the stack's voxel center
            coordinates = (coordinates - (self.stack.vxl_shape - 1) / 2) * self.stack.spacing
        return coordinates
    
    def _init_slice_bbox(self):
        if len(self.coordinates) > 0:
            coords_world = (self.stack.affine_nrmd[:3, :3] @ self.coordinates.T).T + self.stack.affine_nrmd[None, :3, 3]
            self.phys_slice_bbox = np.stack([coords_world.min(axis=0), coords_world.max(axis=0)])
        else:
            self.phys_slice_bbox = None


class DataProcessor(Dataset):
    def __init__(self, args, shuffle=True):
        '''
        Dataset to return non-zero voxel coordinates and values.
        
        Args:
            args: Arguments containing paths and configuration
            shuffle: Whether to shuffle the data/coordinates
        '''
        self.args = args 
        self._init_stacks()
        self._init_coords_and_values()
        self._init_bbox()
        self.normalize_coordinates()
        self.normalize_values()
        # self.normalize_psf_stds()

        # for dataloading
        self.shuffle = shuffle
        self.batch_size = args['batch_size']
        self.indices =  torch.arange(len(self.values)) # indices of the non-zero voxels in the dataset (across stacks)
        self.current_index = 0
        self.steps_per_epoch = len(self.values) // self.batch_size
    
    def _init_stacks(self):
        # Create reference stack
        self.n_slices_global = 0
        self.reference_stack = Stack(self.args, 0, self.n_slices_global)
        self.n_slices_global += self.reference_stack.n_slices
        # Create other stacks with reference to first stack
        self.stacks = [self.reference_stack]
        for i, path in enumerate(self.args['data']['path_stacks'][1:], 1):
            stack = Stack(self.args, i, self.n_slices_global, self.reference_stack)
            self.stacks.append(stack)
            self.n_slices_global += stack.n_slices
        
    def _init_coords_and_values(self):
        # Initialize arrays to store coordinates and values separately
        total_voxels = sum(sum(len(slice_obj.non_zero_values) for slice_obj in stack.slices) for stack in self.stacks)
        self.coords = torch.zeros((total_voxels, 4), dtype=torch.float32)  # x,y,z,slice_id
        self.values = torch.zeros((total_voxels, self.args['sr_out_dim']), dtype=torch.float32)
        self.value_mask = torch.zeros((total_voxels, self.args['sr_out_dim']), dtype=torch.bool) # mask to indicate which values are valid
        self.psf_stds = torch.zeros((total_voxels, 3), dtype=torch.float32)
        self.affine_nrmd = torch.zeros((total_voxels, 4, 4), dtype=torch.float32)
        self.slices_modalities=torch.zeros((self.n_slices_global), dtype=torch.float32)
        # per-slice metadata
        self.slice_meta = {}  # key = global_idx
        idx = 0
        #save_slices_global_order = []
        for i, stack in enumerate(self.stacks):
            #print(f"In the initialisation Stack {i} has {len(stack.slices)} slices")
            for slice_obj in stack.slices:
                out_channel = self.args['data']['output_channel'][i] # output channel for the current stack, i.e., the modality
                self.slices_modalities[slice_obj.global_idx]=out_channel
                #save_slices_global_order.append({
                #"img": slice_obj.img,
                #"stack_id": i,
                #"global_idx": slice_obj.global_idx,
            #})
                n_voxels = len(slice_obj.non_zero_values)
                coords = slice_obj.coordinates
                slice_ids = slice_obj.global_idx * torch.ones((n_voxels, 1)) # slice_ids (global indices of the slices) used as indices for slice_module output!
                psf_stds = stack.psf_stds * torch.ones((n_voxels, 3))
                affine_normalized = torch.from_numpy(stack.affine_nrmd) * torch.ones((n_voxels, 4, 4))
                # Store coordinates and values
                self.coords[idx:idx + n_voxels] = torch.tensor(np.hstack((coords, slice_ids)), dtype=torch.float32)
                self.values[idx:idx + n_voxels, out_channel] = torch.tensor(slice_obj.non_zero_values, dtype=torch.float32)
                self.value_mask[idx:idx + n_voxels, out_channel] = torch.ones((n_voxels), dtype=torch.bool)
                self.psf_stds[idx:idx + n_voxels] = psf_stds
                self.affine_nrmd[idx:idx + n_voxels] = affine_normalized
                #for simulated slices
                self.slice_meta[slice_obj.global_idx] = {
                "start_idx": idx,
                "end_idx": idx + n_voxels,
                "stack_id": i,
                "local_idx": slice_obj.local_idx,
                "global_idx": slice_obj.global_idx,
                "img_shape": slice_obj.img.shape,
                "mask_2d": slice_obj.mask.copy(),
                "non_zero_vxls_2d": slice_obj.non_zero_vxls[:, :2].copy(),  # (x, y) only
                "orig_img": slice_obj.img.copy(),
                "out_channel": int(out_channel),
            }
                idx += n_voxels

        #np.save(os.path.join(self.args['data']['recon_dir'], f'Slices_saved_in_global_order.npy'),np.array(save_slices_global_order,dtype=object))


    def iter_slices(self, batch_size=None):
        """
        Iterate slice by slice using the ORIGINAL coordinates of each slice.
        If batch_size is given, each slice is chunked into smaller voxel batches.
        """
        if batch_size is None:
            batch_size = self.batch_size

        for global_idx in sorted(self.slice_meta.keys()):
            meta = self.slice_meta[global_idx]
            start_idx = meta["start_idx"]
            end_idx = meta["end_idx"]

            n = end_idx - start_idx
            for j in range(0, n, batch_size):
                s = start_idx + j
                e = min(start_idx + j + batch_size, end_idx)

                yield {
                    "global_idx": global_idx,
                    "stack_id": meta["stack_id"],
                    "local_idx": meta["local_idx"],
                    "out_channel": meta["out_channel"],
                    "coords": self.coords[s:e],
                    "values": self.values[s:e],
                    "value_mask": self.value_mask[s:e],
                    "affine_nrmd": self.affine_nrmd[s:e],
                    "psf_stds": self.psf_stds[s:e],
                    "chunk_start_in_slice": j,
                    "chunk_end_in_slice": j + (e - s),
                    "n_voxels_in_slice": n,
                }


    def get_slice_meta(self, global_idx):
        return self.slice_meta[global_idx]


    
    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        '''
        Returns:
            - coords (torch.Tensor): Transformed 3D voxel coordinates with slice and stack id
            - value (torch.Tensor): The intensity value at that voxel
            - psf_stds (torch.Tensor): The standard deviation of the PSF at that voxel
        '''
        coords = self.coords[index]
        value = self.values[index]
        affine_nrmd = self.affine_nrmd[index]
        psf_stds = self.psf_stds[index]
        value_mask = self.value_mask[index]
        return coords, value, value_mask,affine_nrmd, psf_stds

    def __iter__(self):
        ''' Reset the index to 0 and shuffle the indices if shuffle is True '''
        self.current_index = 0
        if self.shuffle:
            self.indices = torch.randperm(len(self.values))
        return self

    def __next__(self):
        ''' Get the next batch '''
        if self.current_index >= len(self.values):
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        coords = self.coords[batch_indices]
        values = self.values[batch_indices]
        value_mask = self.value_mask[batch_indices]
        affine_nrmd = self.affine_nrmd[batch_indices]
        psf_stds = self.psf_stds[batch_indices]
        return coords, values, value_mask, affine_nrmd, psf_stds
    
    def to(self, device):
        self.coords = self.coords.to(device)
        self.values = self.values.to(device)
        self.affine_nrmd = self.affine_nrmd.to(device)
        self.psf_stds = self.psf_stds.to(device)
        self.phys_bbox = self.phys_bbox.to(device)
        self.phys_bbox_size = self.phys_bbox_size.to(device)
        self.slices_modalities = self.slices_modalities.to(device)  # 
        self.value_mask = self.value_mask.to(device)
        return self

    def _init_bbox(self, padding_frac=0.1):
        # get bbox of all stacks using (x,y,z) of the coordinates (world space if use_world_space is True)
        bbox = torch.cat([torch.from_numpy(stack.phys_bbox) for stack in self.stacks], dim=0)
        self.phys_bbox = torch.stack([bbox.min(dim=0).values, bbox.max(dim=0).values]).to(dtype=torch.float32)
        self.phys_bbox_size = self.phys_bbox[1] - self.phys_bbox[0]
        if padding_frac > 0:
            pad_size = self.phys_bbox_size * padding_frac
            self.phys_bbox[0] = self.phys_bbox[0] - pad_size
            self.phys_bbox[1] = self.phys_bbox[1] + pad_size
            self.phys_bbox_size = self.phys_bbox[1] - self.phys_bbox[0]


    def normalize_global_slice_ids(self):
        '''
        Normalize the slice_ids to [-1, 1] for slice module to use
        '''
        slice_min, slice_max = 0, self.n_slices_global-1
        self.slice_ids = 2 * ((self.slice_ids - slice_min) / (slice_max - slice_min)) - 1

    def normalize_coordinates(self):
        '''
        Normalize the coordinates to [-1, 1]
        Don't normalize the slice_ids as we use them as indices for slice_module output!
        '''
        if self.args['data']['normalize_coords']:
            # Normalize coordinates to [-1, 1] using bounding box
            min_, max_ = self.bbox[0], self.bbox[1]
            self.coords[:, :3] = 2 * ((self.coords[:, :3] - min_) / (max_ - min_)) - 1
        
    def normalize_values(self):
      '''
      Normalize the values to [min, max]
      '''
      if self.args['data']['normalize_values']:
          min_, max_ = self.args['data']['normalize_values'][0], self.args['data']['normalize_values'][1]
          if self.args['data']['min_max_values']:
              print("We are using outside normalisation values")
              min_max_values=np.load(self.args['data']['min_max_values'])
          else:
              print("We are normalizing the values")
              min_max_values = np.array([self.values.min(), self.values.max()])
              #print(f"Min: {min_max_values[0]}, Max: {min_max_values[1]}")
              np.save(os.path.join(self.args['data']['recon_dir'], 'min_max_values.npy'), min_max_values)
          #self.values = (self.values - self.values.min()) / (self.values.max() - self.values.min()) * (max_ - min_) + min_
          # Save the min and max values as a .npy file
          self.values = (self.values - min_max_values[0]) / (min_max_values[1] - min_max_values[0]) * (max_ - min_) + min_
          #np.save(os.path.join(self.args['data']['recon_dir'], 'min_max_values.npy'), min_max_values)


    def normalize_psf_stds(self):
        '''
        Normalize the psf_stds: psf_stds are computed from the spacing of the stack, i.e., they are in world space
        and should be normalized to the coordinate space that is in [-1, 1]. 
        In other words, the bbox_size is normalized to 2, so we need to scale the psf_stds by 2/bbox_size
        '''
        if self.args['data']['normalize_coords']:
            self.psf_stds = self.psf_stds * (2 / self.bbox_size)


    def get_full_coordinate_grid(self, device ):
        '''
        Generates a full 3D Cartesian grid of normalized coordinates within the bounding box.

        Returns:
            - coords_flat (torch.Tensor): (N, 3) Cartesian, normalized coordinates of the full grid
            - psf_stds (torch.Tensor): (N, 3) Standard deviation of the PSF at each coordinate
            - recon_shape (tuple): Shape of the full grid (x, y, z) derived from `recon_spacing` and `bbox`
            - affine (torch.Tensor): (4,4) Affine matrix of the reconstructed volume
        '''
        # Compute the shape of the reconstruction grid
        recon_spacing = torch.tensor(self.args['data']['recon_spacing'], device=device) 
        print(f"physique {self.phys_bbox_size}")       
        recon_shape = torch.ceil(self.phys_bbox_size / recon_spacing).long()
        coords_flat, min_, max_ = get_coordinate_grid(recon_shape, self.phys_bbox, device)

        # Compute PSF standard deviations based on the reconstruction spacing
        sigma = 1.2 * recon_spacing / (2 * np.sqrt(2 * np.log(2)))  # In-plane
        sigma = 2 * sigma * (max_ - min_)
        psf_stds = sigma.expand(coords_flat.shape[0], -1)

        ## Define affine transformation matrix
        affine = torch.eye(4, device=device)
        affine[:3, :3] = torch.diag(recon_spacing)  # Scale transformation
        affine[:3, 3] = min_  # Translation (aligning to bbox min)

        return coords_flat, psf_stds, tuple(recon_shape.tolist()), affine
