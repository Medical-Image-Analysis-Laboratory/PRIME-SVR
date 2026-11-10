"""PRIME-SVR reconstruction in the space and on the grid of a reference volume (e.g. a fetal atlas),
from a finished PRIME-SVR run (prime_svr_pipeline.py or run_prime_svr.py).

  1. rigid registration of one reconstructed channel (moving, <recon_dir>/<sub>_recon_<c>.nii.gz) on the reference
     (fixed), ANTs Mattes MI, optionally starting from an initial ITK transform (e.g. a manual ITK-SNAP alignment of
     that same file on the reference); the direction of the initial transform (as is, or inverted) is detected
  2. every reference voxel is mapped to the PRIME-SVR grid with this transform
  3. the trained network (<recon_dir>/svr_model.pth) is evaluated directly at those points, for every channel:
     no resampling of the reconstructed volume, the same transform for every TE

  <out-dir>/<sub>_recon_echo-<c+1>_space-<name>.nii.gz   reference grid, intensities of the input stacks
  <out-dir>/<sub>_recon_mask_space-<name>.nii.gz         with --mask
  <out-dir>/primesvr_to_<name>_0GenericAffine.mat        ANTs transform

  python primesvr_to_reference.py <run>/config.yaml /atlas/STA30.nii.gz --init init.txt --name STA30
  python primesvr_to_reference.py <run>/config.yaml /atlas/STA30.nii.gz --mask <run>/<sub>_recon_mask.nii.gz
"""
import os
import sys
import glob
import argparse
import yaml
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import to_reference, recon_mask  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Config of the PRIME-SVR run (its data.recon_dir holds svr_model.pth)")
    parser.add_argument("reference", help="Reference volume (fixed), e.g. a fetal atlas, brain extracted")
    parser.add_argument("--init", default=None,
                        help="Initial ITK transform (.txt/.mat, e.g. ITK-SNAP) between the registered channel "
                             "<recon_dir>/<sub>_recon_<channel>.nii.gz and the reference")
    parser.add_argument("--channel", type=int, default=0, help="Channel registered on the reference (default 0)")
    parser.add_argument("--name", default="atlas", help="Name of the reference space in the outputs (default atlas)")
    parser.add_argument("--mask", default=None,
                        help="Brain mask on the reconstruction grid (e.g. <sub>_recon_mask.nii.gz of the pipeline), "
                             "applied to the outputs")
    parser.add_argument("--out-dir", default=None, help="Output folder (default <recon_dir>/space-<name>)")
    parser.add_argument("--normalized", action="store_true",
                        help="Keep the network values in [0, 1] instead of the intensities of the input stacks")
    parser.add_argument("--device", default="cuda", help="Torch device (default cuda)")
    parser.add_argument("--batch-size", type=int, default=100000)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    recon_dir = config["data"]["recon_dir"]
    channels = {}
    for p in glob.glob(os.path.join(recon_dir, "*_recon_*.nii.gz")):
        suffix = os.path.basename(p).split("_recon_")[-1][:-len(".nii.gz")]
        if suffix.isdigit():
            channels[int(suffix)] = p
    if args.channel not in channels:
        raise FileNotFoundError(f"No <sub>_recon_{args.channel}.nii.gz in {recon_dir}")
    subject = os.path.basename(channels[args.channel]).split("_recon_")[0]
    out_dir = os.path.abspath(args.out_dir or os.path.join(recon_dir, f"space-{args.name}"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"{subject}: channel {args.channel} -> {args.reference}")
    fixed, moving, transforms = to_reference.register(channels[args.channel], args.reference, out_dir,
                                                      args.name, args.init)
    values, inside = to_reference.sample_network(config, recon_dir, fixed, moving, transforms,
                                                 device=args.device, batch_size=args.batch_size)
    ref_affine = nib.load(args.reference).affine

    keep = inside
    if args.mask:
        mask = np.asanyarray(nib.load(args.mask).dataobj) > 0
        if mask.shape != moving.shape:
            raise ValueError(f"--mask {mask.shape} is not on the reconstruction grid {moving.shape}")
        # same voxel grid as the registered channel, so its header is used (the pipeline mask has another affine)
        keep = inside & to_reference.warp_mask(moving.new_image_like(mask.astype(np.float32)), fixed, transforms)
        mask_img = nib.Nifti1Image(keep.astype(np.uint8), ref_affine)
        mask_img.set_data_dtype(np.uint8)
        path = os.path.join(out_dir, f"{subject}_recon_mask_space-{args.name}.nii.gz")
        nib.save(mask_img, path)
        print(f"  {path}")

    for c in sorted(channels):
        vol = np.zeros(fixed.shape, np.float32)
        vol[inside] = values[:, c] if args.normalized else recon_mask.unnormalize(values[:, c], recon_dir)
        path = os.path.join(out_dir, f"{subject}_recon_echo-{c + 1}_space-{args.name}.nii.gz")
        nib.save(nib.Nifti1Image(vol * keep, ref_affine), path)
        print(f"  {path}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        sys.exit(f"ERROR: {e}")
