"""PRIME-SVR reconstruction in the space and on the grid of a reference volume (e.g. a fetal atlas).

From fqMRI_clean/primesvr_to_reference.py:
  1. rigid registration of one reconstruction (moving) on the reference (fixed), ANTs Mattes MI, optionally
     starting from an initial ITK transform (e.g. a manual ITK-SNAP alignment); the direction of the initial
     transform (as is, or inverted) that matches the reference best is used
  2. every reference voxel is mapped to the PRIME-SVR grid: ANTs moves voxel-index images of the PRIME-SVR grid
     onto the reference grid (linear interpolation of a linear function is exact), so no LPS/RAS bookkeeping
  3. the trained PRIME-SVR network (svr_model.pth) is evaluated directly at those points, for every channel:
     no resampling of the reconstructed volume, the same transform for every TE
"""
import os
import numpy as np
import torch
import ants


def corr(a, b, mask):
    a, b = a[mask], b[mask]
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def load_model(config, recon_dir, device):
    from models.inr_svr import SVR
    config = dict(config, device=device)
    state = torch.load(os.path.join(recon_dir, "svr_model.pth"), map_location=device, weights_only=False)
    n_slices = len(np.load(os.path.join(recon_dir, "mc.npy")))  # only used by the slice module
    model = SVR(config, n_slices, config["epochs"]).to(device)
    # only the SR module is evaluated: its weights are loaded alone, so models trained with an earlier
    # slice module (other layer names) still load
    sr_state = {k[len("sr_module."):]: v for k, v in state.items() if k.startswith("sr_module.")}
    model.sr_module.load_state_dict(sr_state)
    model.eval()
    return model


def register(moving_path, reference, out_dir, ref_name, init=None):
    """Rigid registration of moving_path on reference. Returns (fixed, moving, ANTs transforms)."""
    fixed = ants.image_read(reference)
    moving = ants.image_read(moving_path)
    fixed_np = fixed.numpy()
    ref_mask = fixed_np > np.percentile(fixed_np[fixed_np > 0], 5) if (fixed_np > 0).any() else fixed_np != 0

    initial = None
    if init:
        # ANTs only inverts .mat files: convert (ITK-SNAP writes .txt)
        init_mat = os.path.join(out_dir, "init.mat")
        ants.write_transform(ants.read_transform(init), init_mat)
        scores = {}
        for invert in (False, True):
            warped = ants.apply_transforms(fixed, moving, [init_mat], whichtoinvert=[invert]).numpy()
            scores[invert] = corr(warped, fixed_np, ref_mask)
        invert = max(scores, key=scores.get)
        print(f"  [{ref_name}] init {os.path.basename(init)}: corr as is {scores[False]:.3f}, "
              f"inverted {scores[True]:.3f} -> using {'inverted' if invert else 'as is'}")
        initial = [init_mat]
        if invert:
            initial = [os.path.join(out_dir, "init_inverted.mat")]
            ants.write_transform(ants.read_transform(init_mat).invert(), initial[0])

    reg = ants.registration(fixed, moving, type_of_transform="Rigid", initial_transform=initial,
                            aff_metric="mattes", outprefix=os.path.join(out_dir, f"primesvr_to_{ref_name}_"),
                            verbose=False)
    transforms = reg["fwdtransforms"]
    params = ants.read_transform(transforms[0]).parameters
    angle = np.degrees(np.arccos(np.clip((np.trace(params[:9].reshape(3, 3)) - 1) / 2, -1, 1)))
    print(f"  [{ref_name}] rigid registration: corr {corr(reg['warpedmovout'].numpy(), fixed_np, ref_mask):.3f}, "
          f"rotation {angle:.1f} deg, translation {np.round(params[9:], 1)} mm")
    return fixed, moving, transforms


def sample_network(config, recon_dir, fixed, moving, transforms, device="cuda", batch_size=100000):
    """Network values (N, channels), in [0, 1], at the reference voxels inside the PRIME-SVR grid.
    Returns (values, inside), inside being the (reference shape) boolean array of those voxels."""
    shape = moving.shape
    idx_maps = []
    for axis in range(3):
        grid = np.broadcast_to(np.arange(shape[axis], dtype=np.float32)
                               .reshape([-1 if a == axis else 1 for a in range(3)]), shape)
        img = moving.new_image_like(np.ascontiguousarray(grid))
        idx_maps.append(ants.apply_transforms(fixed, img, transforms, interpolator="linear", defaultvalue=-1).numpy())
    inside = ants.apply_transforms(fixed, moving.new_image_like(np.ones(shape, np.float32)), transforms,
                                   interpolator="linear", defaultvalue=0).numpy() > 0.999
    idx = np.stack(idx_maps, -1)[inside]          # (N, 3) fractional PRIME-SVR voxel index
    u = -1 + 2 * idx / (np.array(shape) - 1)      # PRIME-SVR network input, see utils.transform_utils.get_coordinate_grid

    model = load_model(config, recon_dir, device)
    lo, hi = config["data"]["normalize_values"]
    values = np.zeros((len(u), config["sr_out_dim"]), np.float32)
    coords = torch.tensor(u, dtype=torch.float32, device=device)
    with torch.no_grad():
        for s in range(0, len(coords), batch_size):
            v = model.inference(coords[s:s + batch_size], None, psf_card=0)  # as SVR_Trainer.reconstruct
            values[s:s + batch_size] = ((v.clamp(lo, hi) - lo) / (hi - lo)).cpu().numpy()
    return values, inside


def warp_mask(mask, fixed, transforms):
    """Reconstruction mask (path or ANTs image) moved onto the reference grid (nearest neighbour)."""
    mask = ants.image_read(mask) if isinstance(mask, str) else mask
    warped = ants.apply_transforms(fixed, mask, transforms, interpolator="nearestNeighbor")
    return warped.numpy() > 0.5
