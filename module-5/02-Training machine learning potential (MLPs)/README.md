# Deep Modeling for Molecular Simulation
Hands-on sessions - Day 2 - July 12, 2023

Train your first first-principles machine-learning force field

## Aims

Using the DFT energies and forces obtained in the previous tutorial, train a model for the potential energy surface (PES) using DeePMD-kit.

## Objectives

The objectives of this tutorial session are:
- Train a DeePMD model for the potential energy surface of silicon
- Prepare DFT output data for the training process
- Become familiar with the inputs and outputs of the training process
- Run molecular dynamics simulations driven by the DeePMD model
- Identify strengths and limitations of a rudimentary model
- Use an ensemble of models to estimate the errors in the forces

## Prerequisites

It is assumed that the student has completed all hands-on sessions from [day 1](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1) of this workshop.

## Theory

### Model

In the DeePMD model that we will train for the PES of silicon, the total energy $E$ of a configuration of $N$ atoms with atomic coordinates $\mathbf{R}$ is written as a sum over per-atom energies $E_i$, i.e,

$E(\mathbf{R})=\sum\limits_{i=1}^N E_i = \sum\limits_{i=1}^N E^{\alpha_i}(\mathbf{R}_i)$

where $\mathbf{R}_i$ are the relative atomic coordinates of $N_i$ neighbors in an environment with cutoff $r_c$ around atom $i$, $\alpha_i$ is the atom type of atom $i$, and $E^{\alpha_i}$ is an energy function for atoms of the chemical species $\alpha_i$.
In order to preserve the natural symmetries of the problem, i.e., rotation and permutation of atoms of the same type, we define a vector of descriptors $\mathbf{D}_i$ for atom $i$.
Then, the energy of a configuration can be written as,

$E(\mathbf{R})=\sum\limits_{i=1}^N E^{\alpha_i}(\mathbf{D}_i)$

The starting point for the definition of the descriptors $\mathbf{D}_i$ is a continuous and differentiable switching function,

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq1.png" width="350">
</p>

where $u=(r - r_s)/(r_c - r_s)$, and $r_s$ and $r_c$ are smooth and hard cutoffs, respectively.
Next, we construct a matrix $\mathbf{R}_i \in \mathbb{R}^{N_i \times 4}$ of generalized coordinates with rows,

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq2.png" width="350">
</p>

where $(x_{ij},y_{ij},z_{ij})$ is the distance vector from atom $j$ to atom $i$, and $r_{ij}$ is the norm of such distance.
Furthermore, we define an embedding matrix $\mathbf{G}^i \in \mathbb{R}^{N_i \times M_1}$ with row $j$ given by,  

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq3.png" width="200">
</p>


where $g^{\alpha_i,\alpha_j}$ is a function that maps a scalar into $M_1$ outputs, and is different for each pair of chemical species $\alpha_i$ and $\alpha_j$.
We also define a secondary embedding matrix $\mathbf{G}'^i\in\mathbb{R}^{N_i\times M_2}$ with the first $M_2< M_1$ columns of $\mathbf{G}^i$.

With these ingredients, we now write the descriptor matrix $\mathbf{D}_i \in \mathbb{R}^{M_1 \times M_2}$ as,

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq4.png" width="200">
</p>

which is subsequently flatten into a vector of $M_1 \times M_2$ elements and is used as input in the equation above.
In our simulations, we will use a model for a single species, namely, Si.
$E^{\alpha_i}$ will be represented by a neural network with three layers and 80 neurons per layer, and $g^{\alpha_i,\alpha_j}$ will be represented by a three-layer neural network with sizes 20, 40 and 80, respectively.
Other parameters of our model are $M_1=80$, $M_2=16$, $r_s=3$ Angstrom, and $r_c=6$ Angstrom.

### Loss function

The parameters in the neural networks $E^{\alpha_i}$ and $g^{\alpha_i,\alpha_j}$ described above are determined through the minimization of the following loss function,
    
$\mathcal{L} = \frac{1}{N_\mathcal{B}} \left (\sum_{l \in \mathcal{B}}  \frac{w_{\epsilon}}{N_l} \left | E_l- E(\mathbf{R}^l)\right |^2  + \frac{w_{f}}{3N_l}  \left \| \mathbf{F}_l- \mathbf{F}(\mathbf{R}^l) \right \|^2  \right)$

where $\mathcal{B}$ is a mini-batch (i.e., a subset of the training set) with $N_\mathcal{B}$ atomic configurations,  $w_{\epsilon}$ and $w_{f}$ are weights. 
Furthermore, $E_l$ and $F_l$ are reference energies and forces, $E(\mathbf{R}^l)$ and $\mathbf{F}(\mathbf{R}^l)=-\boldsymbol\nabla_\mathbf{R} E(\mathbf{R}^l)$ are the energy and force predictions of our model for configuration $l$ in the minibatch, and $\mathbf{R}^l$ and $N_l$ are the atomic coordinates and the number of atoms in configuration $l$.

### Optimizer

We will train the models using the Adam optimizer with learning rate $\alpha(i)=\alpha_0 \lambda^{i/\tau}$ where $\alpha_0=0.002$ is the initial learning rate, $\lambda=0.97$, $\tau=5\times10^3$, and $i$ is the step number.
The batch size $N_\mathcal{B}$ is set to one and we will train for a total number of steps equal to $2\times 10^5$.
$w_{\epsilon}$ and $w_{f}$ were varied according to $w_{\epsilon}(i)=w_{\epsilon}^1+(w_{\epsilon}^0-w_{\epsilon}^1)\lambda^{i/\tau}$ and $w_f(i)=w_f^1+(w_f^0-w_f^1)\lambda^{i/\tau}$, with $w_{\epsilon}^0=0.02$, $w_{\epsilon}^1=1$, $w_f^0=1000$, and $w_f^1=1$.
This scheme gives a higher weight to the force term in the loss function at the beginning of the training process, and by the end of it both the energy and force term have equal weights.

## Training data

We have practiced how to prepare the training data, so now we will use the training data for silicon prepared on [hand-on session 3](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1/3-preparing-training-data).
These data consist of:
- Around 400 configurations of silicon in the cubic diamond crystal structure obtained using random displacements from equilibrium atomic positions. 
- Around 300 configurations of liquid silicon at 1700 K obtained in molecular dynamics simulations driven by the [Stillinger-Weber potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262).

## Training process

An example input script for the training process is provided in ```scripts/input.json```.
Before executing it, let's analyze its contents.
The first block is the model definition:
```json
   "model": {
        "type_map":     ["Si"],
        "descriptor": {
            "type": "se_a",
            "sel": [30],
            "rcut_smth": 3.0,
            "rcut": 6.0,
            "neuron": [
                20,
                40,
                80
            ],
            "axis_neuron": 16,
            "seed": 25875,
        },
        "fitting_net": {
            "neuron": [
                80,
                80,
                80
            ],
            "resnet_dt": true,
            "seed": 25875,
        },
```
We will discuss this input in the classroom and you can also find further information [here](https://docs.deepmodeling.com/projects/deepmd/en/master/model/train-se-e2-a.html).

The next two blocks specify options for the optimization process and the definition of the loss function:
```json
    "learning_rate": {
        "start_lr": 0.002,
        "decay_steps": 500,
    },
    "loss": {
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
    },
```

In the last block we will specify, among other things, the training and validation data:
```json
    "training": {
        "stop_batch": 200000,
        "disp_file": "lcurve.out",
        "disp_freq": 2000,
        "save_freq": 20000,
        "save_ckpt": "model.ckpt",
        "validation_data": {
            "systems": [
                "<SOME_FOLDER>/perturbations-si-64/0.05A-2p"
            ],
            "batch_size":       "auto"
        },
        "training_data": {
            "systems": [
                "<SOME_FOLDER>/perturbations-si-64/0.01A-1p",
                "<SOME_FOLDER>/perturbations-si-64/0.1A-3p",
                "<SOME_FOLDER>/perturbations-si-64/0.2A-5p",
                "<SOME_FOLDER>/liquid-si-64/trajectory-lammps-1700K-1bar/extracted-confs",
                "<SOME_FOLDER>/liquid-si-64/trajectory-lammps-1700K-10000bar/extracted-confs",
                "<SOME_FOLDER>/liquid-si-64/trajectory-lammps-1700K-neg10000bar/extracted-confs"
                        ],
            "batch_size":       "auto"
        }
    }
```

Edit <SOME_FOLDER> to point to the directory with your training data.
Note that we have chosen a somewhat arbitrary separation between training and validation data.

Now it's time to start training the model for the potential energy surface!
Execute ```dp train input.json``` to start training.
The training process should take about 15 minutes.
You can monitor its progress in the file lcurve.out.
The first few lines are as follows:
```
#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn         lr
      0      1.96e+01    6.35e+00      9.76e-01    9.36e-01      6.20e-01    1.98e-01    2.0e-03
   2000      2.64e+00    1.06e+00      6.86e-02    7.54e-02      8.84e-02    3.47e-02    1.8e-03
   4000      1.39e+00    7.93e+00      3.97e-02    4.74e-02      4.94e-02    2.83e-01    1.6e-03
   6000      2.19e+00    7.75e+00      4.27e-02    1.04e-01      8.28e-02    2.94e-01    1.4e-03
   8000      1.24e+00    7.51e+00      4.99e-02    8.79e-02      4.89e-02    3.02e-01    1.2e-03
  10000      1.06e+00    5.91e+00      6.94e-02    4.24e-02      4.23e-02    2.53e-01    1.1e-03
```
where the columns represent the training steps, the total RMS error (val-validation and trn-training), the RMS error in energy, the RMS error in the forces, and the learning rate.
You can plot the number of steps vs the RMS errors to follow the progress of the training process.

Training MLPs using DeepMD-kit
-----------------------------

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/02-Training%20machine%20learning%20potential%20%28MLPs%29/training_mlps.ipynb)

Compressinging MLPs using DeepMD-kit
-----------------------------

Once the training is complete, we can proceed to freeze the model using ```dp freeze```.
This will create a deep potential file ```frozen_model.pb``` that can be used for inference (running MD or simply computing energies/forces).
It is useful to [compress](https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00102) the model using ```dp compress -t input.json -i frozen_model.pb -o frozen_model_compressed.pb```.
This will create a model ```frozen_model_compressed.pb``` that can perform inference significantly faster than ```frozen_model.pb```.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/02-Training%20machine%20learning%20potential%20%28MLPs%29/compressing_mlps.ipynb)

> **Note** Deep neural network potentials can achieve high accuracy, but often at the cost of large model sizes and high computational demands. Compressing MLPs helps reduce model complexity and inference time without significantly compromising accuracy. This is particularly useful for large-scale molecular dynamics simulations, where efficiency and scalability are critical. In this notebook, we demonstrate how to compress trained models using DeepMD-kit.



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
