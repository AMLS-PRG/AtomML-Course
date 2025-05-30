# Build Large Structures for MLP-driven MD Simulations

Now that we have trained a machine learning potential (MLP), we are able to perform molecular dynamics simulations at a level of accuracy comparable to ab initio molecular dynamics (AIMD), but at a fraction of the computational cost.
This enables us to simulate systems much larger than those tractable by DFT-based methods like Quantum ESPRESSO (QE).

Here, we will build a new LAMMPS data file containing the expanded model, supercell_3x3x3.data:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/04-Performing%20MD%20simulations/small-data/build_large_structures.ipynb)


