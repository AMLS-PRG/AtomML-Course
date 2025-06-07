# Analyzing the Trajectory

Herein, we prepared two trajectories for solid and liquid silicon systems.
The total simulation time is 100 ps, with snapshots saved every 0.1 ps, resulting in 1000 frames in total.

We can initially preview the trajectories online.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Check_trj.ipynb)
Does it show the expected behavior for a solid?

Also, you can now copy the trajectories ```si.lammps-dump-text```[*Click here*](https://drive.google.com/drive/folders/1L7pvvC1_ZBMy2mbndwKh-rO2q6pRYRCG?usp=drive_link) and ```si_solid.lammps-dump-text```[*Click here*](https://drive.google.com/file/d/1JCFXgqru7S-DCJIZO2Xvar87Tdgfx3qZ/view?usp=drive_link) to your laptop and visualize it with Ovito. 

Once that you have loaded the LAMMPS dump file into Ovito, you can color the atoms according to the degree of order around them.
Apply the ```Identify diamond structure``` modifier that can be chosen from the ```Add modification``` dropdown menu.
For reference, below we show liquid and solid configurations colored with the modifier ```Identify diamond structure```.

<p float="left">
  <img src="https://github.com/PabloPiaggi/Crystallization-of-Silicon/raw/master/si-liquid.png" width="250"> 
  <img src="https://github.com/PabloPiaggi/Crystallization-of-Silicon/raw/master/si-solid.png"  width="250">
</p>

1. Radial Distribution Function (RDF) ！！！！！！！！！！！！！！！！！wiki 
Purpose:
To quantify how atoms are spatially distributed relative to each other.

Insight:
RDF reveals short- and long-range ordering, distinguishing liquid-like disorder from solid crystalline structure.

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Calculate_rdf.ipynb)

2. Mean Square Displacement (MSD)
Purpose:
To measure atomic mobility over time.

Insight:
MSD is used to estimate diffusion and compare atomic motion in different phases (e.g., solid vs. liquid).

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Calculate_MSD.ipynb)

3. Z-Axis Density Profile
Purpose:
To evaluate how atom density varies along the z-axis.

Insight:
This helps detect structural layering.

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Calculate_density.ipynb)
