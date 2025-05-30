# Trained Machine Learning Potentials (MLPs)

This folder contains **four trained machine learning potentials (MLPs)** as well as the **loss curve file (`lcurve.out`)** from the training process of one of the potentials.

## Contents

- `frozen_model_1_compressed.pb`, `frozen_model_2_compressed.pb`, `frozen_model_3_compressed.pb`, `frozen_model_4_compressed.pb`  
  → Fully trained MLP models.

- `lcurve.out`  
  → Training log file that records the evolution of the **loss function**, including the RMSE of energy and forces on both training and validation sets.

## 🔍 Loss Curve Visualization

To visualize the training loss evolution of the model:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/03-Trained%20MLPs%20(Folder%2002%20Output)/checking_lcurve_out.ipynb)


---

These MLPs enable molecular dynamics simulations.
