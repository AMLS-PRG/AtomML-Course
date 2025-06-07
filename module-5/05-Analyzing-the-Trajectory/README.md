# Analyzing the Trajectory

## Visual trajectory
Herein, we prepared two trajectories for solid and liquid silicon systems.
The total simulation time is 100 ps, with snapshots saved every 0.1 ps, resulting in 1000 frames in total.

We can initially preview the trajectories online.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Check_trj.ipynb)
Does it show the expected behavior for a solid?

Also, you can now copy the trajectories ```si.lammps-dump-text```[*Click here*](https://drive.google.com/file/d/1CSkQOyYbQ0aHf67ETLJ58Gp5RlnPkSxZ/view?usp=drive_link) and ```si_solid.lammps-dump-text```[*Click here*](https://drive.google.com/file/d/1JCFXgqru7S-DCJIZO2Xvar87Tdgfx3qZ/view?usp=drive_link) to your laptop and visualize it with Ovito. 

Once that you have loaded the LAMMPS dump file into Ovito, you can color the atoms according to the degree of order around them.
Apply the ```Identify diamond structure``` modifier that can be chosen from the ```Add modification``` dropdown menu.
For reference, below we show liquid and solid configurations colored with the modifier ```Identify diamond structure```.

<p float="left">
  <img src="https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/05-Analyzing-the-Trajectory/si-liquid.png" width="250"> 
  <img src="https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/05-Analyzing-the-Trajectory/si-solid.png"  width="250">
</p>

## Z-Axis density profile

To evaluate how atom density varies along the z-axis.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Calculate_density.ipynb)

## Radial distribution function (RDF)

To quantify how atoms are spatially distributed relative to each other.
RDF reveals short- and long-range ordering, distinguishing liquid-like disorder from solid crystalline structure.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Calculate_rdf.ipynb)

## Mean square displacement (MSD)

To measure atomic mobility over time.
MSD is used to estimate diffusion and compare atomic motion in different phases (e.g., solid vs. liquid).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/05-Analyzing-the-Trajectory/Calculate_MSD.ipynb)

