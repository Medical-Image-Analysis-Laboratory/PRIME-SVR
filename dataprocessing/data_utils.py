import numpy as np
import nibabel.processing as nip
import scipy.ndimage as ndi
import subprocess
import os
import fsl.data.image as fimage
import fsl.transform.flirt as flirt
import nibabel as nib
import ants
from scipy.ndimage import affine_transform
from dipy.align.imaffine import transform_centers_of_mass, MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D


GA_VOL_DICT = {
    "21": 96822.93,
    "22": 108088.93,
    "23": 113467.99,
    "24": 137928.18,
    "25": 188990.77,
    "26": 211795.16,
    "27": 221781.17,
    "28": 244230.75,
    "29": 268206.09,
    "30": 294281.12,
    "31": 324412.21,
    "32": 352803.53,
    "33": 367171.73,
    "34": 404292.1,
    "35": 430329.25,
    "36": 439290.24,
    "37": 446248.8,
    "38": 457769.27}


def align_by_center_of_mass(rec_img_nii, rec_mask_nii, ref_mask_nii, stack_id=None, path_output=None):
    # Get the affine matrices for the input images
    rec_aff = rec_mask_nii.affine
    ref_aff = ref_mask_nii.affine

    # Load the mask data into arrays
    rec_mask_arr = rec_mask_nii.get_fdata()
    ref_mask_arr = ref_mask_nii.get_fdata()

    # Get the coordinates of the non-zero voxels
    vxls_non_zero_rec = np.argwhere(rec_mask_arr)
    world_coords_rec = nib.affines.apply_affine(rec_aff, vxls_non_zero_rec) 
    vxls_non_zero_ref = np.argwhere(ref_mask_arr)
    world_coords_ref = nib.affines.apply_affine(ref_aff, vxls_non_zero_ref)

    # Compute the center of mass for each mask (mean of the non-zero voxel coordinates)
    com_rec = np.mean(world_coords_rec, axis=0)
    com_ref = np.mean(world_coords_ref, axis=0)

    # Compute the translation required to align the centers of mass
    translation = com_ref - com_rec

    # Update the affine matrix with the new translation
    rec_aff[:3, 3] += translation  # Apply translation in the first three columns

    # Create new NIfTI images with the updated affine matrices
    rec_img_nii = nib.Nifti1Image(rec_img_nii.get_fdata(), rec_aff)
    rec_mask_nii = nib.Nifti1Image(rec_mask_nii.get_fdata(), rec_aff)
    
    if stack_id is not None:
        # save in a tmp path
        tmp_path = f"{path_output}/aligned_{stack_id}.nii.gz" 
        tmp_path2 = f"{path_output}/aligned_mask_{stack_id}.nii.gz"
        #nib.save(rec_img_nii, tmp_path)
        #nib.save(rec_mask_nii, tmp_path2)
    return rec_img_nii, rec_mask_nii

