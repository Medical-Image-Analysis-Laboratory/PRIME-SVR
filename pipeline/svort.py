"""SVoRT motion correction (nesvor register) and its conversion into PRIME-SVR inputs.

PRIME-SVR then starts from the SVoRT slice transforms, in the SVoRT atlas space: every SVoRT slice becomes a
single-slice "stack" carrying the SVoRT affine (align_stacks: none, clip_percentile: null).

Only the transforms of the SVoRT slices are used: nesvor rescales every stack by its own 99th percentile, which
would remove the contrast between TEs. The intensities are taken from the input stacks, clipped stack-wise as
PRIME-SVR does. nesvor keeps the input stack order and, inside each stack, the slices whose mask (stack mask and
image > 0) is not empty, in z order; each slice is saved with its first axis flipped (the affine accounts for it).
The mapping is checked on the voxel support of every slice.
"""
import os
import json
import subprocess
import numpy as np
import nibabel as nib

NESVOR_IMAGE = "junshenxu/nesvor:v0.5.0"
CLIP_PERCENTILE = 99.9  # PRIME-SVR default, applied on the whole stack


def run_svort(stacks, masks, thicknesses, out_dir, mount_root, gpu=0, image=NESVOR_IMAGE):
    """nesvor register --registration svort on the stacks; every path must be under mount_root."""
    def to_container(path):
        rel = os.path.relpath(os.path.abspath(path), mount_root)
        if rel.startswith(".."):
            raise ValueError(f"{path} is not under {mount_root}")
        return "/data/" + rel

    os.makedirs(os.path.join(out_dir, "slices"), exist_ok=True)
    out = to_container(out_dir)
    cmd = ["docker", "run", "--rm", "--gpus", f"device={gpu}", "--ipc=host",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-v", f"{os.path.abspath(mount_root)}:/data", image, "nesvor", "register",
           "--input-stacks", *map(to_container, stacks),
           "--stack-masks", *map(to_container, masks),
           "--thicknesses", *map(str, thicknesses),
           "--registration", "svort",
           "--output-slices", f"{out}/slices",
           "--output-json", f"{out}/svort.json",
           "--output-log", f"{out}/svort.log"]
    print(f"  [svort] nesvor register on {len(stacks)} stacks (atlas space)")
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    log = os.path.join(out_dir, "svort.log")
    if result.returncode != 0:
        raise RuntimeError(f"SVoRT failed, see {log}\n{result.stderr[-2000:]}")
    for line in open(log):
        if "similarity score" in line or "transformation" in line:
            print("  [svort] " + line.split("] ", 1)[-1].strip())


def svort_slices(stacks, masks, brain_masks, thicknesses, channels, svort_dir):
    """Single-slice PRIME-SVR inputs from the nesvor register outputs in svort_dir (run on `stacks`, `masks`).

    Writes <svort_dir>/primesvr_slices/<stack>_slice-<z>_{T2w,mask,brainmask}.nii.gz and returns the lists
    (path_stacks, path_masks, brain_masks, slice_thickness, output_channel), one entry per slice."""
    svort = json.load(open(os.path.join(svort_dir, "svort.json")))
    if len(svort["input_stacks"]) != len(stacks):
        raise ValueError(f"{len(svort['input_stacks'])} SVoRT stacks for {len(stacks)} stacks")
    in_slices = os.path.join(svort_dir, "slices")
    out_slices = os.path.join(svort_dir, "primesvr_slices")
    os.makedirs(out_slices, exist_ok=True)

    out = {"path_stacks": [], "path_masks": [], "brain_masks": [], "slice_thickness": [], "output_channel": []}
    k = 0
    for stack_path, mask_path, brain_path, thickness, channel in zip(stacks, masks, brain_masks, thicknesses, channels):
        stack = nib.load(stack_path).get_fdata(dtype=np.float32)
        mask = (np.asanyarray(nib.load(mask_path).dataobj) > 0).reshape(stack.shape)
        brain = (np.asanyarray(nib.load(brain_path).dataobj) > 0).reshape(stack.shape)
        clip = np.percentile(stack, CLIP_PERCENTILE)
        prefix = os.path.basename(stack_path).split("_desc-")[0]
        for z in [z for z in range(stack.shape[2]) if (mask[..., z] & (stack[..., z] > 0)).any()]:
            svort_slice = nib.load(os.path.join(in_slices, f"{k}.nii.gz"))
            img = stack[::-1, :, z]
            if not np.array_equal(svort_slice.get_fdata()[..., 0] > 0, img > 0):
                raise ValueError(f"SVoRT slice {k} does not match slice {z} of {stack_path}")
            base = os.path.join(out_slices, f"{prefix}_slice-{z:02d}")
            header = svort_slice.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(np.clip(img, 0, clip)[..., None], svort_slice.affine, header), base + "_T2w.nii.gz")
            header.set_data_dtype(np.uint8)
            for suffix, m in (("_mask", mask), ("_brainmask", brain)):
                nib.save(nib.Nifti1Image(m[::-1, :, z, None].astype(np.uint8), svort_slice.affine, header),
                         base + suffix + ".nii.gz")
            out["path_stacks"].append(base + "_T2w.nii.gz")
            out["path_masks"].append(base + "_mask.nii.gz")
            out["brain_masks"].append(base + "_brainmask.nii.gz")
            out["slice_thickness"].append(thickness)
            out["output_channel"].append(channel)
            k += 1
    if os.path.isfile(os.path.join(in_slices, f"{k}.nii.gz")):
        raise ValueError(f"More SVoRT slices than expected ({k}) in {in_slices}")
    print(f"  [svort] {k} slices -> {out_slices}")
    return out


def use_svort_slices(data, slices):
    """Point the data section of a PRIME-SVR config to the SVoRT slices."""
    for key in ("path_stacks", "path_masks", "slice_thickness", "output_channel"):
        data[key] = slices[key]
    data["align_stacks"] = "none"     # the slices are already aligned by SVoRT
    data["clip_percentile"] = None    # already clipped stack-wise
    data["simulated_slices"] = False  # would write one simulated stack per slice
