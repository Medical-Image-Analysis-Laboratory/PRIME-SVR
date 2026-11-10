"""Stack/mask matching and preprocessing (denoising, brain extraction, mask dilation) before PRIME-SVR."""
import os
import re
import json
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_fill_holes

# name parts that only describe the kind of file, ignored when matching a stack with its mask
IGNORED_TOKENS = {"t2w", "t2", "mask", "masks", "brain", "brainmask", "dilated", "desc", "denoised",
                  "extracted", "seg", "bet", "label", "nii", "gz"}


def nifti_stem(path):
    name = os.path.basename(path)
    return re.sub(r"\.nii(\.gz)?$", "", name)


def list_niftis(paths):
    """A folder (all its .nii/.nii.gz files) or a list of files."""
    if len(paths) == 1 and os.path.isdir(paths[0]):
        files = glob.glob(os.path.join(paths[0], "*.nii")) + glob.glob(os.path.join(paths[0], "*.nii.gz"))
    else:
        files = paths
    files = sorted(os.path.abspath(f) for f in files)
    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f)
    if not files:
        raise FileNotFoundError(f"No NIfTI file in {paths}")
    return files


def match_key(path):
    tokens = re.split(r"[_\-.]+", nifti_stem(path).lower())
    return tuple(t for t in tokens if t and t not in IGNORED_TOKENS)


def match_masks(stacks, masks):
    """Pair every stack with one mask: same name once the kind-of-file parts (T2w, mask, desc-brain, ...) are removed.
    e.g. sub-01_run-1_T2w.nii.gz <-> sub-01_run-1_desc-brain_mask.nii.gz or sub-01_run-1_mask.nii.gz
    Stacks without a mask are skipped with a warning; several possible masks for one stack is an error."""
    mask_keys = {m: match_key(m) for m in masks}
    pairs, errors = [], []
    for stack in stacks:
        key = match_key(stack)
        candidates = [m for m, k in mask_keys.items() if k == key]
        if not candidates:  # e.g. extra prefix/suffix on one side: the shorter key must be a contiguous part
            candidates = [m for m, k in mask_keys.items() if _contains(k, key) or _contains(key, k)]
        if len(candidates) > 1:  # prefer the undilated mask, it is dilated here
            undilated = [m for m in candidates if "dilated" not in os.path.basename(m).lower()]
            candidates = undilated or candidates
        if len(candidates) == 1:
            pairs.append((stack, candidates[0]))
        elif not candidates:
            print(f"  WARNING: no mask for {os.path.basename(stack)}, skipped")
        else:
            found = ", ".join(os.path.basename(c) for c in candidates) or "none"
            errors.append(f"  {os.path.basename(stack)}: {len(candidates)} masks ({found})")
    used = [m for _, m in pairs]
    if len(set(used)) != len(used):
        errors.append("  the same mask was matched to several stacks: " +
                      ", ".join(sorted({os.path.basename(m) for m in used if used.count(m) > 1})))
    if not pairs and not errors:
        errors.append("  no stack has a mask")
    if errors:
        raise ValueError("Could not match the stacks with the masks:\n" + "\n".join(errors) +
                         "\nGive the masks the stack names (e.g. <stack>_mask.nii.gz).")
    return pairs


def _contains(longer, shorter):
    n = len(shorter)
    return n > 0 and any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def sidecar(path):
    json_path = re.sub(r"\.nii(\.gz)?$", ".json", path)
    return json.load(open(json_path)) if os.path.isfile(json_path) else {}


def echo_time(path):
    """TE in ms from the JSON sidecar (EchoTime, s) or the NIfTI description ("TE=82;..."), else None."""
    te = sidecar(path).get("EchoTime")
    if te is not None:
        return float(te) * 1000 if te < 10 else float(te)
    descrip = nib.load(path).header["descrip"].tobytes().decode(errors="ignore")
    match = re.search(r"TE=([0-9.eE+-]+)", descrip)
    return float(match.group(1)) if match else None


def slice_thickness(path):
    """Slice thickness in mm from the JSON sidecar, else the slice spacing of the NIfTI header."""
    thickness = sidecar(path).get("SliceThickness")
    return float(thickness) if thickness else float(nib.load(path).header.get_zooms()[2])


def preprocess_stack(stack, mask, out_dir, denoise=True, dilate_iter=4, overwrite=False):
    """denoise (ANTs non-local means, Gaussian noise) -> brain extraction with the mask -> dilated mask.
    Outputs keep the header of the input stack. Returns the paths of the extracted stack and the masks."""
    os.makedirs(out_dir, exist_ok=True)
    stem = nifti_stem(stack)
    out = {
        "extracted": os.path.join(out_dir, f"{stem}_desc-extracted_T2w.nii.gz"),
        "brain_mask": os.path.join(out_dir, f"{stem}_desc-brain_mask.nii.gz"),
        "dilated_mask": os.path.join(out_dir, f"{stem}_desc-brainDilated_mask.nii.gz"),
    }
    if not overwrite and all(os.path.isfile(p) for p in out.values()):
        print(f"  [preprocess] exists, skipping: {stem}")
        return out

    img = nib.load(stack)
    shape = img.shape[:3]
    mask_img = nib.load(mask)
    if mask_img.shape[:3] != shape:
        raise ValueError(f"{os.path.basename(mask)} {mask_img.shape} does not have the shape of "
                         f"{os.path.basename(stack)} {img.shape}")
    brain = (np.asanyarray(mask_img.dataobj) > 0).reshape(shape)
    if not brain.any():
        raise ValueError(f"Empty mask: {mask}")

    if denoise:
        import ants  # same denoising as SirenSVR/denoise_fetal.py; ANTs and nibabel share the voxel order
        data = ants.denoise_image(image=ants.image_read(stack), noise_model="Gaussian").numpy()
    else:
        data = img.get_fdata(dtype=np.float32)
    data = np.asarray(data, dtype=np.float32).reshape(shape)

    header = img.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(data * brain, img.affine, header), out["extracted"])
    # same dilation as SirenSVR/dilate_mask.py: fill holes, then binary dilation
    dilated = binary_dilation(binary_fill_holes(brain), iterations=dilate_iter)
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), img.affine, header), out["brain_mask"])
    nib.save(nib.Nifti1Image(dilated.astype(np.uint8), img.affine, header), out["dilated_mask"])
    print(f"  [preprocess] {stem}: {'denoised, ' if denoise else ''}extracted, mask dilated ({dilate_iter} it.)")
    return out
