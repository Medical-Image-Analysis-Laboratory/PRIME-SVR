# PRIME-SVR
 Physics-infoRmed Implicit Multi-Echo Slice-to-Volume Reconstruction for Fetal T2 mapping 

 
## Overview
**PRIME-SVR** is the first implicit neural representation (INR) framework for **joint slice to volume reconstruction from multi-echo stacks of fetal MRI**. The method is **fully self-supervised**: no ground-truth HR volume, no external motion estimation network, and no manually annotated training data are required.

## Method
![PRIME-SVR pipeline overview](prime_svr_overview.jpeg)

In PRIME-SVR, we model the multi-echo HR intensities as an implicit continuous function defined over 3D spatial coordinates, $\mathbf{V}$. The multi-echo data observed in the acquired 2D slices are treated as sparse, discrete, and degraded samples of this underlying function. The degradation process is slice-specific and modeled through a slice acquisition model, where a subset of its parameters is estimated by a separate implicit continuous function $\mathbf{f_{SM}}$ defined over the coordinates within the slice. Both functions are parameterized by $\theta$ and $\theta'$, respectively, and are learned in a self-supervised manner for each subject by minimizing the discrepancy between the observed slices and those simulated through the slice acquisition model. In addition, a regularization term derived from the Bloch equations is imposed on $\mathbf{V_\theta}$ to model the physical relationship across TEs. Once learned, $\mathbf{V_\theta}$ is queried on any grid sampling the 3D volume, %to reconstruct the HR multi-echo volumes, 
from which a T2 map is estimated.

## Installation

## Data preparation

PRIME-SVR expects, per subject:

- 3 stacks (sagittal, axial, coronal) acquired at **N_TE = 3** echo times (extendable to other values),
- brain masks for each stack,
Optimal used il faut run le precoess de bias field, et denoised before running.
- Echo times

**Recommended preprocessing before running PRIME-SVR:**

- **Denoising** of every stack (e.g. non-local means, as implemented in ANTs).
- **Bias field correction**, run slice-wise on each stack.

Both steps are handled outside the main pipeline and should be applied to the raw stacks beforehand 

## Start

Reconstruction is driven by a single YAML configuration file, passed to the main training/reconstruction script:

```bash
python run_prime_svr.py --config configs/config.yaml
```

## Citation

If you use PRIME-SVR in your research, please cite:

```bibtex
@article{bulut2025primesvr,
  title   = {Physics-Informed Joint Multi-TE Super-Resolution with Implicit Neural Representation for Robust Fetal T2 Mapping},
  author  = {Bulut, Busra and Dannecker, Maik and Sanchez, Thomas and Neves Silva, Sara and Jia, Steven and Ledoux, Jean-Baptiste and Pomar, Leo and Sichitiu, Joanna and Gomez, Yvan and Koob, Meriam and Dunet, Vincent and Deprez, Maria and Auzias, Guillaume and Rousseau, Fran\c{c}ois and Hutter, Jana and Rueckert, Daniel and Bach Cuadra, Meritxell},
  journal = {arXiv preprint arXiv:2508.10680},
  year    = {2026}
}
```

## Contact

For questions, please open an issue or contact **Busra Bulut** — busra.bulut@unil.ch — Department of Radiology, Lausanne University Hospital and University of Lausanne.
