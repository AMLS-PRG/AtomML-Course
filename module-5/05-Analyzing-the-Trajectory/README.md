# Analyzing the Trajectory

You can now copy the trajectory ```si.lammps-dump-text``` to your laptop and visualize it with Ovito. 
https://drive.google.com/drive/folders/1L7pvvC1_ZBMy2mbndwKh-rO2q6pRYRCG?usp=drive_link

Does it show the expected behavior for a solid?

Once that you have loaded the LAMMPS dump file into Ovito, you can color the atoms according to the degree of order around them.
Apply the ```Identify diamond structure``` modifier that can be chosen from the ```Add modification``` dropdown menu.
For reference, below we show liquid and solid configurations colored with the modifier ```Identify diamond structure```.

<p float="left">
  <img src="https://github.com/PabloPiaggi/Crystallization-of-Silicon/raw/master/si-liquid.png" width="250"> 
  <img src="https://github.com/PabloPiaggi/Crystallization-of-Silicon/raw/master/si-solid.png"  width="250">
</p>

You can also plot thermodynamic properties of the system that have been printed to the file ```thermo.txt```.

Next, let's analyze the contents of the file ```md.out```, which should be similar to:
```
#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
           0       6.141403e-03       6.360340e-07       3.476040e-03       6.160679e-03       1.564173e-03       3.614160e-03
        1000       3.983807e-03       3.534347e-04       2.002133e-03       2.265209e-02       5.212460e-03       1.369438e-02
        2000       4.729147e-03       3.986826e-04       2.127513e-03       1.862923e-02       5.243879e-03       1.081513e-02
        3000       6.839780e-03       2.654651e-04       3.016102e-03       2.869552e-02       5.306404e-03       1.235991e-02
```
We suggest that you plot steps (column 1) vs the maximum deviation of the forces (column 5), and the steps (column 1) vs the average deviation of the forces (column 7).
Are the value of the errors stable? What are their magnitudes? Can you conclude that the model is well-trained to describe the solid, or does it require further training?

Now that we have studied the performance of our rudimentary model for the solid, let's run a simulation for liquid silicon.
An appropriate LAMMPS file is provided in ```molecular-dynamics/liquid/input.lmp```.
The simulation uses a thermostat and barostat to mantain a temperature of 1700 K (our guess for the melting temperature) and a pressure of 1 bar.
Once the simulation has completed, analyze it with the same steps described above for the solid.
Is the model suitable to describe liquid silicon? Why?

In [hands-on session 5](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2/5-active-learning) you will learn a technique to systematically improve the stability and accuracy of the models.


1. Radial Distribution Function (RDF)
Purpose:
To quantify how atoms are spatially distributed relative to each other.

Insight:
RDF reveals short- and long-range ordering, distinguishing liquid-like disorder from solid crystalline structure.

2. Mean Square Displacement (MSD)
Purpose:
To measure atomic mobility over time.

Insight:
MSD is used to estimate diffusion and compare atomic motion in different phases (e.g., solid vs. liquid).

3. Z-Axis Density Profile
Purpose:
To evaluate how atom density varies along the z-axis.

Insight:
This helps detect structural layering, surface effects, or melting near interfaces.
