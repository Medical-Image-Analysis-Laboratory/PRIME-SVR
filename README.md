# PRIME-SVR
 Physics-infoRmed Implicit Multi-Echo Slice-to-Volume Reconstruction for Fetal T2 mapping 
 
## Overview

## Method
![PRIME-SVR pipeline overview](pipeline_overview.jpeg)

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

## Run the command 

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
