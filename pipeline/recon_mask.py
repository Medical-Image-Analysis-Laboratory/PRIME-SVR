"""Brain mask, original intensities and world affine of a PRIME-SVR reconstruction.

Mask: every input slice is mapped into the reconstruction grid exactly as PRIME-SVR maps it, including the motion
correction it learned (mc.npy): world = affine_nrmd (R(mc) c + t(mc)), c = centred voxel coordinates in mm.
Each slice votes on the voxels it covers (in-plane pixel and slice thickness): brain if the voxel is in its brain
(undilated) mask, background if it is only in its dilated mask. A voxel is brain when more than `threshold` of
its votes are brain; holes are then closed/filled and the largest component kept.

Grid: PRIME-SVR samples the padded bounding box of all slice mask voxels with linspace, so recon voxel i is at
world coordinate bbox[0] + i * step. The affine PRIME-SVR saves is centred on the grid instead; world_affine()
gives the affine in the space of the input stacks (scanner space, or SVoRT atlas space).
"""
import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import binary_closing, binary_fill_holes, label

from utils.transform_utils import euler2rot  # same rotation convention as PRIME-SVR

BBOX_PADDING = 0.1  # DataProcessor._init_bbox


def load_stacks(data, brain_masks):
    """Stack masks and affines as PRIME-SVR uses them (including the center of mass alignment)."""
    stacks = []
    for i, mask_path in enumerate(data["path_masks"]):
        mask_img = nib.load(mask_path)
        shape = nib.load(data["path_stacks"][i]).shape[:3]
        mask = (mask_img.get_fdata() > 0).reshape(shape)
        brain = (nib.load(brain_masks[i]).get_fdata() > 0).reshape(shape) & mask
        affine = mask_img.affine.copy()
        if data["align_stacks"] == "center_of_mass" and i > 0:
            # dataprocessing.data_utils.align_by_center_of_mass, on the reference (first) stack mask
            com = nib.affines.apply_affine(affine, np.argwhere(mask)).mean(0)
            affine[:3, 3] += stacks[0]["com"] - com
        stacks.append({
            "mask": mask, "brain": brain, "affine": affine,
            "spacing": np.array(mask_img.header.get_zooms()[:3]),
            "com": nib.affines.apply_affine(affine, np.argwhere(mask)).mean(0),
            "thickness": data["slice_thickness"][i],
        })
    return stacks


def to_world(stack, vxls, mc=None):
    """Voxel indices (N, 3) of a stack -> world coordinates, after the slice motion mc (6,) if given."""
    shape = np.array(stack["mask"].shape)
    coords = (vxls - (shape - 1) / 2) * stack["spacing"]  # Slice._process_coordinates (use_world_space False)
    if mc is not None:
        R = euler2rot(torch.tensor(mc[:3], dtype=torch.float64)).numpy()
        coords = coords @ R.T + mc[3:]
    # affine_nrmd maps the centred mm coordinates to world space, i.e. world = affine @ voxel
    A = stack["affine"]
    A_nrmd = A[:3, :3] @ np.diag(1 / stack["spacing"])
    return coords @ A_nrmd.T + (A[:3, 3] + A[:3, :3] @ ((shape - 1) / 2))


def recon_grid(stacks, recon_spacing, recon_shape):
    """bbox (2, 3) and voxel step of the PRIME-SVR reconstruction grid, checked against the recon shape."""
    world = np.concatenate([to_world(s, np.argwhere(s["mask"])) for s in stacks])
    bbox = np.stack([world.min(0), world.max(0)])
    pad = (bbox[1] - bbox[0]) * BBOX_PADDING
    bbox = np.stack([bbox[0] - pad, bbox[1] + pad])
    shape = np.array(recon_shape[:3])
    expected = np.ceil((bbox[1] - bbox[0]) / np.array(recon_spacing)).astype(int)
    if not np.array_equal(shape, expected):
        raise ValueError(f"recon shape {tuple(shape)} != {tuple(expected)} computed from the config")
    return bbox, (bbox[1] - bbox[0]) / (shape - 1)


def world_affine(bbox, step):
    affine = np.eye(4)
    affine[:3, :3] = np.diag(step)
    affine[:3, 3] = bbox[0]
    return affine


def brain_mask(data, brain_masks, recon_dir, recon_shape, threshold=0.5):
    """Brain mask of the reconstruction (bool array) and the world affine of its grid."""
    mc_all = np.load(os.path.join(recon_dir, "mc.npy"))
    stacks = load_stacks(data, brain_masks)
    # slices as PRIME-SVR numbers them: stack order, then z, only slices with mask voxels
    slices = [(s, z) for s in stacks for z in range(s["mask"].shape[2]) if s["mask"][..., z].any()]
    if len(slices) != len(mc_all):
        raise ValueError(f"{len(slices)} slices in the config but {len(mc_all)} in mc.npy")
    bbox, step = recon_grid(stacks, data["recon_spacing"], recon_shape)
    shape = np.array(recon_shape[:3])

    # sub-voxel samples of a slice voxel: 3x3 in plane, 5 through the slice thickness
    u = np.linspace(-1 / 3, 1 / 3, 3)
    brain_votes = np.zeros(shape, np.float32)
    all_votes = np.zeros(shape, np.float32)
    for (stack, z), mc in zip(slices, mc_all):
        w = np.linspace(-0.5, 0.5, 5) * stack["thickness"] / stack["spacing"][2]
        offsets = np.stack(np.meshgrid(u, u, w, indexing="ij"), -1).reshape(-1, 3)
        for votes, mask in ((all_votes, stack["mask"]), (brain_votes, stack["brain"])):
            ij = np.argwhere(mask[..., z])
            if len(ij) == 0:
                continue
            vxls = np.column_stack([ij, np.full(len(ij), z)])
            pts = (vxls[:, None, :] + offsets[None]).reshape(-1, 3)
            idx = np.rint((to_world(stack, pts, mc) - bbox[0]) / step).astype(int)
            idx = idx[((idx >= 0) & (idx < shape)).all(1)]
            np.add.at(votes, tuple(idx.T), 1)

    mask = (all_votes > 0) & (brain_votes > threshold * all_votes)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    labels, n = label(mask)
    if n > 1:
        mask = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    print(f"  [mask] brain mask: {mask.sum() * np.prod(data['recon_spacing']) / 1000:.1f} cm3 ({len(slices)} slices)")
    return mask, world_affine(bbox, step)


def unnormalize(recon, recon_dir):
    """PRIME-SVR output in [0, 1] -> intensities of the input stacks (min_max_values.npy of the run).
    The output is (v - min) / (max - min) whatever data.normalize_values is."""
    vmin, vmax = np.load(os.path.join(recon_dir, "min_max_values.npy"))
    return recon * (vmax - vmin) + vmin
