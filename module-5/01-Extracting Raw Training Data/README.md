Get Raw Files using dpdata
-----------------------------
This notebook demonstrates how to extract raw training data (positions, energies, forces, cell, and atom types) from QE .out files using the dpdata library.
It is suitable for converting multiple converged QE outputs into a DeepMD-ready raw dataset.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Extracting%20Raw%20Training%20Data/get_raw_files_dpdata.ipynb)



Get Raw Files using ASE
-----------------------------
This notebook demonstrates how to extract and process raw data manually using the ase library, including handling per-frame energies, forces, atomic positions, and custom type mapping.
Recommended when fine control or error handling is required for specific QE .out structures.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Extracting%20Raw%20Training%20Data/get_raw_files_ASE.ipynb)

