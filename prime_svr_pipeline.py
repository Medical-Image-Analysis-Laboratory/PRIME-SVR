"""PRIME-SVR from a folder of T2w stacks and a folder of brain masks, in one command.

  1. pairs every stack with its mask by name (sub-01_run-1_T2w.nii.gz <-> sub-01_run-1_mask.nii.gz, ..._desc-brain_mask, ...)
  2. preprocessing: denoising (ANTs non-local means), brain extraction with the mask, mask dilation
  3. PRIME-SVR (one output channel, or one per TE with --TEs)
  4. brain mask of the reconstruction from the input slice masks (with the learned motion), and the
     reconstruction in the intensities of the input stacks, in the space of the input stacks

  <output>/<sub>_recon[_echo-<n>].nii.gz   reconstruction, brain masked, input intensities (echo n = n-th of --TEs)
  <output>/<sub>_recon_mask.nii.gz         brain mask
  <output>/config.yaml                     PRIME-SVR config of the run
  <output>/before_postprocessing/          PRIME-SVR recon_dir: raw outputs and trained model, deleted at the
                                           end unless --keep-before-postprocessing
  <output>/preprocessed/, svort/           preprocessed stacks and SVoRT slices, deleted at the end
                                           unless --keep-intermediate

  python prime_svr_pipeline.py --stacks <T2w dir> --masks <mask dir> --output <dir>
  python prime_svr_pipeline.py --stacks <T2w dir> --masks <mask dir> --output <dir> --svort --epochs 300
  python prime_svr_pipeline.py --stacks <T2w dir> --masks <mask dir> --output <dir> --svort --no-motion-correction
  python prime_svr_pipeline.py --stacks <T2w dir> --masks <mask dir> --output <dir> --TEs 308 397 598 --coeff 1
"""
import os
import re
import sys
import glob
import argparse
import yaml
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import preprocess  # noqa: E402
from pipeline import svort  # noqa: E402

TE_TOLERANCE = 0.15  # relative tolerance between the TE of a stack and the TE of a channel


def default_config():
    return {
        "data": {
            "path_stacks": [], "path_masks": [], "slice_thickness": [], "output_channel": [],
            "align_stacks": "center_of_mass",
            "bias_field_correction": "slice_wise",
            "normalize_coords": False,
            "normalize_values": [0, 1],
            "min_max_values": "",
            "use_world_space": False,
            "recon_spacing": [0.8, 0.8, 0.8],
            "recon_dir": "",
            "simulated_slices": False,
            "coeff": 0,
            "echotime": [],
        },
        "use_amp": True, "compile": False, "device": "cuda",
        "psf_card_start": 1, "psf_card_end": 64, "psf_card_schedule": "quadratic",
        "batch_size": 25000, "epochs": 60, "load_model": False, "model_path": "",
        "sm_in_dim": 1, "sm_out_dim": 8, "sm_n_units": 128, "sm_n_hidden_layers": 3, "sm_omega_0": 30,
        "sm_last_linear": True, "sm_omega_schedule": "vanilla",
        "outlier_rejection": True, "motion_correction": True,
        "sr_model": "siren", "sr_loss_metric": "l1", "sr_in_dim": 3, "sr_out_dim": 1,
        "sr_n_units": 330, "sr_n_hidden_layers": 6, "sr_omega_0": 30, "sr_last_linear": True,
        "sr_omega_schedule": "vanilla",
        "lr_sr": 5e-5, "lr_sm": 5e-4, "scheduler": "cosine", "step_size": 30, "gamma": 0.1,
    }


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def set_option(config, assignment):
    """--set key=value, key may be nested with dots (data.recon_spacing=[0.5,0.5,0.5])."""
    key, _, value = assignment.partition("=")
    if not value:
        raise ValueError(f"--set expects key=value, got {assignment}")
    *parents, leaf = key.split(".")
    node = config
    for p in parents:
        node = node[p]
    if leaf not in node:
        raise KeyError(f"--set: unknown config key {key}")
    node[leaf] = yaml.safe_load(value)


def assign_channels(stacks, tes):
    """Output channel of every stack: 0 without --TEs, else the TE of its header/sidecar."""
    if not tes:
        return [0] * len(stacks)
    channels = []
    for stack in stacks:
        te = preprocess.echo_time(stack)
        if te is None:
            raise ValueError(f"No TE found for {os.path.basename(stack)} (JSON EchoTime or 'TE=' in the header)")
        closest = min(range(len(tes)), key=lambda i: abs(tes[i] - te))
        channels.append(closest if abs(tes[closest] - te) <= TE_TOLERANCE * tes[closest] else None)
    return channels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stacks", nargs="+", required=True, help="Folder of T2w stacks (or the stack files)")
    parser.add_argument("--masks", nargs="+", required=True, help="Folder of brain masks (or the mask files)")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--TEs", nargs="+", type=float, default=None,
                        help="One output channel per TE, stacks assigned by the TE of their header/JSON "
                             "(default: one channel 0 for all stacks)")
    parser.add_argument("--coeff", type=float, default=0, help="Bloch regularization weight, needs --TEs (default 0)")
    parser.add_argument("--epochs", type=int, default=60, help="PRIME-SVR epochs (default 60)")
    parser.add_argument("--thickness", type=float, default=None,
                        help="Slice thickness in mm (default: JSON SliceThickness, else the slice spacing)")
    parser.add_argument("--spacing", type=float, default=0.8, help="Isotropic reconstruction spacing in mm (default 0.8)")
    parser.add_argument("--svort", action="store_true", help="SVoRT motion correction before PRIME-SVR (atlas space)")
    parser.add_argument("--no-denoise", action="store_true", help="Skip the denoising")
    parser.add_argument("--no-motion-correction", action="store_true",
                        help="Disable PRIME-SVR's slice motion correction (slices stay where they are, e.g. after "
                             "--svort); slice scaling and outlier rejection are kept")
    parser.add_argument("--dilate-iter", type=int, default=4, help="Mask dilation iterations (default 4)")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Fraction of slices voting brain for the reconstruction mask (default 0.5)")
    parser.add_argument("--mask-dilate", type=int, default=3,
                        help="Dilation of the final reconstruction mask in voxels, to avoid cropping the brain "
                             "(default 3, 0 to disable)")
    parser.add_argument("--reference", default=None,
                        help="Reference volume (e.g. a fetal atlas): the reconstruction is also given in its space "
                             "and on its grid, by evaluating the trained network at its voxels")
    parser.add_argument("--reference-init", default=None,
                        help="Initial ITK transform for the registration on --reference (.txt/.mat, e.g. ITK-SNAP)")
    parser.add_argument("--reference-name", default="atlas",
                        help="Name of the reference space in the output names (default atlas)")
    parser.add_argument("--sr-model", choices=["siren", "multi_siren"], default="siren")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Any other PRIME-SVR config option, e.g. --set batch_size=20000 --set data.align_stacks=none")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute every step (default: reuse existing outputs)")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Keep the preprocessed stacks (preprocessed/) and the SVoRT slices (svort/); "
                             "by default they are deleted at the end")
    parser.add_argument("--keep-before-postprocessing", action="store_true",
                        help="Keep before_postprocessing/ (raw PRIME-SVR outputs and trained model), needed to run "
                             "primesvr_to_reference.py afterwards or to reuse the training; by default it is deleted "
                             "at the end")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before torch is imported
    if args.coeff != 0 and not args.TEs:
        sys.exit("ERROR: --coeff (Bloch regularization) needs --TEs")
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------ 1. stacks and masks
    pairs = preprocess.match_masks(preprocess.list_niftis(args.stacks), preprocess.list_niftis(args.masks))
    channels = assign_channels([s for s, _ in pairs], args.TEs)
    tes = args.TEs or [preprocess.echo_time(pairs[0][0]) or 0]
    print(f"{len(pairs)} stacks:")
    selected = []
    for (stack, mask), channel in zip(pairs, channels):
        thickness = args.thickness or preprocess.slice_thickness(stack)
        if channel is None:
            print(f"  skip {os.path.basename(stack)} (TE {preprocess.echo_time(stack):g} not in --TEs)")
            continue
        print(f"  {os.path.basename(stack)} <- {os.path.basename(mask)} | channel {channel} | {thickness:g} mm")
        selected.append((channel, stack, mask, thickness))
    for c, te in enumerate(tes):
        if not any(s[0] == c for s in selected):
            sys.exit(f"ERROR: no stack for channel {c} (TE {te:g})")
    # ordered by channel; the first stack is PRIME-SVR's reference stack
    selected.sort(key=lambda s: (s[0], natural_key(s[1])))

    # ------------------------------------------------ 2. preprocessing
    print("Preprocessing")
    prep = [preprocess.preprocess_stack(stack, mask, os.path.join(out_dir, "preprocessed"),
                                        denoise=not args.no_denoise, dilate_iter=args.dilate_iter,
                                        overwrite=args.overwrite)
            for _, stack, mask, _ in selected]

    config = default_config()
    data = config["data"]
    data["path_stacks"] = [p["extracted"] for p in prep]
    data["path_masks"] = [p["dilated_mask"] for p in prep]
    data["slice_thickness"] = [s[3] for s in selected]
    data["output_channel"] = [s[0] for s in selected]
    data["recon_spacing"] = [args.spacing] * 3
    data["recon_dir"] = os.path.join(out_dir, "before_postprocessing")
    data["coeff"] = args.coeff
    data["echotime"] = tes
    config["epochs"] = args.epochs
    config["sr_out_dim"] = len(tes)
    config["sr_model"] = args.sr_model
    config["motion_correction"] = not args.no_motion_correction
    brain_masks = [p["brain_mask"] for p in prep]

    # ------------------------------------------------ 3. SVoRT
    if args.svort:
        print("SVoRT")
        svort_dir = os.path.join(out_dir, "svort")
        if os.path.isfile(os.path.join(svort_dir, "svort.json")) and not args.overwrite:
            print(f"  [svort] exists, skipping registration: {svort_dir}")
        else:
            svort.run_svort(data["path_stacks"], data["path_masks"], data["slice_thickness"], svort_dir,
                            mount_root=out_dir, gpu=args.gpu)
        slices = svort.svort_slices(data["path_stacks"], data["path_masks"], brain_masks,
                                    data["slice_thickness"], data["output_channel"], svort_dir)
        svort.use_svort_slices(data, slices)
        brain_masks = slices["brain_masks"]

    for assignment in args.set:
        set_option(config, assignment)
    config_path = os.path.join(out_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config: {config_path}")
    if data["coeff"] != 0:
        print(f"Bloch regularization is used for multi-TE reconstruction (coeff = {data['coeff']:g}, "
              f"TEs = {', '.join(f'{te:g}' for te in tes)} ms)")

    # ------------------------------------------------ 4. PRIME-SVR
    recon_dir = data["recon_dir"]
    os.makedirs(recon_dir, exist_ok=True)
    recons = {c: sorted(glob.glob(os.path.join(recon_dir, f"*_recon_{c}.nii.gz"))) for c in range(len(tes))}
    # the reconstruction is reused only if it was made with the same config
    used_config = os.path.join(recon_dir, "config_used.yaml")
    same_config = os.path.isfile(used_config) and yaml.safe_load(open(used_config)) == config
    if all(recons.values()) and same_config and not args.overwrite:
        print(f"PRIME-SVR: exists, skipping training ({recon_dir})")
    else:
        for old in glob.glob(os.path.join(recon_dir, "*_recon_*.nii.gz")):
            os.remove(old)
        print(f"PRIME-SVR: {len(data['path_stacks'])} {'slices' if args.svort else 'stacks'}, {args.epochs} epochs")
        from sv_reconstruction import SVR_Trainer
        trainer = SVR_Trainer(config)
        trainer.train()
        if data.get("simulated_slices"):
            trainer.reconstruct_simulated_stacks(save_prefix="simulated_stack")
        trainer.save_model(os.path.join(recon_dir, "svr_model.pth"))
        with open(used_config, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        recons = {c: sorted(glob.glob(os.path.join(recon_dir, f"*_recon_{c}.nii.gz"))) for c in range(len(tes))}

    # ------------------------------------------------ 5. mask and original intensities
    print("Brain mask and outputs")
    from pipeline import recon_mask
    first = nib.load(recons[0][0])
    mask, affine = recon_mask.brain_mask(data, brain_masks, recon_dir, first.shape, args.mask_threshold)
    if args.mask_dilate > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=args.mask_dilate)
        print(f"  [mask] dilated by {args.mask_dilate} voxels: "
              f"{mask.sum() * np.prod(data['recon_spacing']) / 1000:.1f} cm3")
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    mask_img.set_data_dtype(np.uint8)
    # subject name as PRIME-SVR wrote it: <subject>_recon_<channel>.nii.gz
    subject = os.path.basename(recons[0][0]).split("_recon_")[0]
    mask_path = os.path.join(out_dir, f"{subject}_recon_mask.nii.gz")
    nib.save(mask_img, mask_path)
    print(f"  {subject}_recon_mask.nii.gz")
    names = []
    for c, te in enumerate(tes):
        values = recon_mask.unnormalize(nib.load(recons[c][0]).get_fdata(dtype=np.float32), recon_dir)
        name = f"{subject}_recon" + (f"_echo-{c + 1}" if args.TEs else "")
        nib.save(nib.Nifti1Image(values * mask, affine), os.path.join(out_dir, name + ".nii.gz"))
        print(f"  {name}.nii.gz" + (f" (TE {te:g} ms)" if args.TEs else ""))
        names.append(name)

    # ------------------------------------------------ 6. reference (atlas) space
    if args.reference:
        from pipeline import to_reference
        space = args.reference_name
        print(f"Reference space: {space} ({args.reference})")
        space_dir = os.path.join(out_dir, f"space-{space}")
        os.makedirs(space_dir, exist_ok=True)
        # the first echo, masked, is registered; the same transform is used for every echo
        fixed, moving, transforms = to_reference.register(os.path.join(out_dir, names[0] + ".nii.gz"),
                                                          args.reference, space_dir, space, args.reference_init)
        values, inside = to_reference.sample_network(config, recon_dir, fixed, moving, transforms)
        ref_mask = inside & to_reference.warp_mask(mask_path, fixed, transforms)
        ref_affine = nib.load(args.reference).affine
        mask_ref_img = nib.Nifti1Image(ref_mask.astype(np.uint8), ref_affine)
        mask_ref_img.set_data_dtype(np.uint8)
        nib.save(mask_ref_img, os.path.join(out_dir, f"{subject}_recon_mask_space-{space}.nii.gz"))
        print(f"  {subject}_recon_mask_space-{space}.nii.gz")
        for c, name in enumerate(names):
            vol = np.zeros(fixed.shape, np.float32)
            vol[inside] = recon_mask.unnormalize(values[:, c], recon_dir)
            nib.save(nib.Nifti1Image(vol * ref_mask, ref_affine), os.path.join(out_dir, f"{name}_space-{space}.nii.gz"))
            print(f"  {name}_space-{space}.nii.gz")

    # ------------------------------------------------ intermediate files
    # not needed afterwards: primesvr_to_reference.py only uses config.yaml and before_postprocessing/
    import shutil
    to_delete = []
    if not args.keep_intermediate:
        to_delete += [("preprocessed", "--keep-intermediate"), ("svort", "--keep-intermediate")]
    if not args.keep_before_postprocessing:
        to_delete.append((os.path.basename(recon_dir), "--keep-before-postprocessing"))
    for folder, option in to_delete:
        path = os.path.join(out_dir, folder)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted {folder}/ ({option} to keep it)")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        sys.exit(f"ERROR: {e}")
