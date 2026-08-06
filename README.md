# PRIME-SVR
 Physics-infoRmed Implicit Multi-Echo Slice-to-Volume Reconstruction for Fetal T2 mapping 
 
## Overview
## Method


## Installation


## Data preparation

PRIME-SVR expects, per subject:

- 3 stacks (sagittal, axial, coronal) acquired at **N_TE = 3** echo times (extendable to other values),
- brain masks for each stack (e.g. from [FET-BET](https://github.com/IntelligentImaging/fetal-brain-extraction)),
Optimal used il faut run le precoess de bias field, et denoised before running.
- Echo times

## Run the command 


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
