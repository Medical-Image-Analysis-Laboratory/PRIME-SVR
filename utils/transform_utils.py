import torch
import numpy as np


def embed2affine(embed):
    """Convert embedding to rotation matrix and translation vector.
    Args:
        embed (torch.Tensor): (..., 6) tensor containing euler angles and translation
    Returns:
        R (torch.Tensor): (..., 3, 3) rotation matrix
        t (torch.Tensor): (..., 3) translation vector
    """
    R = euler2rot(embed[..., :3])
    t = embed[..., 3:]
    return R, t


def euler2rot(theta):
    """Convert euler angles to rotation matrix using vectorized operations.
    Args:
        theta (torch.Tensor): (..., 3) tensor containing euler angles
    Returns:
        R (torch.Tensor): (..., 3, 3) rotation matrix
    """
    # Pre-compute sin and cos values once
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    
    # Unpack for better readability
    s1, s2, s3 = sin[..., 0], sin[..., 1], sin[..., 2]
    c1, c2, c3 = cos[..., 0], cos[..., 1], cos[..., 2]
    
    # Create output tensor directly
    R = torch.empty(theta.shape[:-1] + (3, 3), dtype=theta.dtype, device=theta.device)
    
    # Fill the rotation matrix directly - avoid intermediate tensors
    R[..., 0, 0] = c1*c3 - c2*s1*s3
    R[..., 0, 1] = -c1*s3 - c2*c3*s1
    R[..., 0, 2] = s1*s2
    R[..., 1, 0] = c3*s1 + c1*c2*s3
    R[..., 1, 1] = c1*c2*c3 - s1*s3
    R[..., 1, 2] = -c1*s2
    R[..., 2, 0] = s2*s3
    R[..., 2, 1] = c3*s2
    R[..., 2, 2] = c2
    
    return R


def get_psf_stds(spacing, slice_thickness=None):
        '''
        Compute the standard deviation of the PSF based on the spacing of the stack.
        If slice_thickness is not provided, the last entry of the spacing is used.
        If all spacing entries are the same and slice_thickness is not provided, the PSF is isotropic.
        '''
        if slice_thickness is None:
            slice_thickness = spacing[2]
        isotropic = np.allclose(spacing[:2], spacing[2])
        z_factor = 1.2 if isotropic else 1.0
        denom = 2 * np.sqrt(2 * np.log(2))
        sigma_xy = 1.2 * np.array(spacing[:2]) / denom  # In-plane
        sigma_z = z_factor * slice_thickness / denom  # Slice thickness
        return torch.tensor([sigma_xy[0], sigma_xy[1], sigma_z])


def get_coordinate_grid(recon_shape, bbox, device, normalize_coords=True, flatten=True):
    # Generate 3D coordinate grid
    x = torch.linspace(bbox[0, 0], bbox[1, 0], recon_shape[0], device=device)
    y = torch.linspace(bbox[0, 1], bbox[1, 1], recon_shape[1], device=device)
    z = torch.linspace(bbox[0, 2], bbox[1, 2], recon_shape[2], device=device)
    
    # Create meshgrid for Cartesian coordinates
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")  # Shape: (X_dim, Y_dim, Z_dim)

    # Flatten coordinate grid into (N, 3)
    coords_flat = torch.stack([X, Y, Z], dim=-1) 
    if flatten:
        coords_flat = coords_flat.view(-1, 3)  # (N, 3)

    if normalize_coords:
        # **Center coordinates by the coordinate grid's center**
        coords_flat = coords_flat - torch.mean(coords_flat, dim=0)

    min_ = coords_flat.min(dim=0).values
    max_ = coords_flat.max(dim=0).values
    coords_flat = 2 * ((coords_flat - min_) / (max_ - min_)) - 1
    return coords_flat, min_, max_

def torch_hom_matrix(R, t, t_first=False):
    hom_matrix = torch.eye(4).to(torch.float).cuda(0)
    hom_matrix[:3, :3] = R
    hom_matrix[:3, 3] = R @ t if t_first else t
    return hom_matrix

def torch_inv_hom_matrix(hom_matrix):
    R = hom_matrix[:3, :3]
    translation = hom_matrix[:3, 3]
    inv_R = torch.inverse(R)
    inv_translation = -torch.matmul(inv_R, translation)
    inv_hom_matrix = torch.eye(4).to(torch.float).cuda(0)
    inv_hom_matrix[:3, :3] = inv_R
    inv_hom_matrix[:3, 3] = inv_translation
    return inv_hom_matrix