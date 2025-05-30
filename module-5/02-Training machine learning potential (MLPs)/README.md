### 🔍 Before You Train the MLPs: Explore the Dataset
-----------------------------
Before training the MLPs, you can explore the dataset in LAMMPS trajectory (`.trj`) format (A trajectory that does not reach the accuracy of first-principles calculations.).

This trajectory will be used as input for Quantum ESPRESSO (QE) to calculate atomic **coordinates**, **forces**, **system energy**, and **virial** — which will then form the final **dataset** for training the machine learning potentials.

📌 You can:

- Inspect the structure and atomic configuration in each frame.
- Observe the **size of the simulation cell/model** in this trajectory.
- Later, **compare it with the model size** used in final MD simulations to better understand transferability and scaling.

This step helps you build intuition about the dataset before diving into model training.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/02-Training%20machine%20learning%20potential%20%28MLPs%29/check_data.ipynb)

Training MLPs using DeepMD-kit
-----------------------------

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/02-Training%20machine%20learning%20potential%20%28MLPs%29/training_mlps.ipynb)

Compressinging MLPs using DeepMD-kit
-----------------------------
Deep neural network potentials can achieve high accuracy, but often at the cost of large model sizes and high computational demands. Compressing MLPs helps reduce model complexity and inference time without significantly compromising accuracy. This is particularly useful for large-scale molecular dynamics simulations, where efficiency and scalability are critical. In this notebook, we demonstrate how to compress trained models using DeepMD-kit.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/02-Training%20machine%20learning%20potential%20%28MLPs%29/compressing_mlps.ipynb)


