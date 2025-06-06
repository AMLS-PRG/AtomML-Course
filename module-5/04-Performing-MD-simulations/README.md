# MD Simulations Driven by the MLP

## Build Large Structures for MLP-driven MD Simulations

Now that we have trained a machine learning potential (MLP), we are able to perform molecular dynamics simulations at a level of accuracy comparable to ab initio molecular dynamics (AIMD), but at a fraction of the computational cost.
This enables us to simulate systems much larger than those tractable by DFT-based methods like Quantum ESPRESSO (QE).

Here, we will build a new LAMMPS data file containing the expanded model, supercell_3x3x3.data:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/04-Performing-MD-simulations/small-data/build_large_structures.ipynb)

## Running molecular dynamics simulations

We will now run molecular dynamics simulations.
A LAMMPS script (input_local.lmp) to simulate liquid silicon can be found at ```module-5/04-Performing-MD-simulations/runLammps```.
This input file has been annotated to help you understand the purpose of each line.
The simulation uses a thermostat and barostat to mantain a temperature of 1700 K and a pressure of 1 bar.

The lines of the input file that instruct the code to use the DeePMD model is:
```
pair_style      deepmd ./frozen_model_1_compressed.pb ./frozen_model_2_compressed.pb ./frozen_model_3_compressed.pb ./frozen_model_4_compressed.pb out_file md.out out_freq ${out_freq}
pair_coeff      * *
```
where ```frozen_model_?_compressed.pb``` are four models trained on the same data and different initial random seeds.
These four models are employed to estimate the errors in the forces.
We define the error $\epsilon_i$ in the $i$-th force component as $\epsilon_i^2 = \langle | f_i-\bar{f}_i |^2 \rangle$, where $\bar{f}_i = \langle f_i \rangle$ and the average $\langle \cdot \rangle$ is taken over the ensemble of models.
The average, minimum, and maximum errors in the forces are reported every ```out_freq``` steps in the file ```md.out```.
Four models are provided in ```module-5/04-Performing-MD-simulations/runLammps/frozen_model_?_compressed.pb```, but you are encourage to use your own model trained in the previous section.
Also, you may want to share models with other participants.

You can now download all the necessary input files and run the molecular dynamics simulation on a GPU-enabled local cluster such as Hyperion at DIPC [(Visit DIPC Hyperion Documentation)](https://scc.dipc.org/docs/).

📥 Download input files:
```
wget https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/04-Performing-MD-simulations/runLammps/input_local.lmp
wget https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/04-Performing-MD-simulations/runLammps/frozen_model_1_compressed.pb
wget https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/04-Performing-MD-simulations/runLammps/frozen_model_2_compressed.pb
wget https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/04-Performing-MD-simulations/runLammps/frozen_model_3_compressed.pb
wget https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/04-Performing-MD-simulations/runLammps/frozen_model_4_compressed.pb
wget https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/04-Performing-MD-simulations/runLammps/liquid_supercell_3x3x3.data
wget https://raw.githubusercontent.com/AMLS-PRG/AtomML-Course/main/module-5/04-Performing-MD-simulations/runLammps/runlammps.sbatch
```
🚀 Submit the job to Hyperion:
```
sbatch runlammps.sbatch
```
You can also run a molecular dynamics simulation for a solid silicon system.
This simulation uses a thermostat and barostat to maintain a temperature of 300 K and a pressure of 1 bar.

Download the same input files (input_local.lmp), MLP models (*.pb), and sbatch script (*.sbatch) as described above (***Place it in a separate folder from the liquid simulation.***).

Additionally, download the solid-state data file:
solid_supercell_3x3x3.data

Modify the input file (input_local.lmp):
Update the temperature and the data file name as shown below:
```
variable        temperature equal 300.0 # Target temperature in K
read_data      ./solid_supercell_3x3x3.data
```
Once you've made the changes, submit the updated *.lmp file again using the same batch script.


In the simulation process, you can check the ```md.out``` file., which should be similar to:
```
#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
           0       6.141403e-03       6.360340e-07       3.476040e-03       6.160679e-03       1.564173e-03       3.614160e-03
        1000       3.983807e-03       3.534347e-04       2.002133e-03       2.265209e-02       5.212460e-03       1.369438e-02
        2000       4.729147e-03       3.986826e-04       2.127513e-03       1.862923e-02       5.243879e-03       1.081513e-02
        3000       6.839780e-03       2.654651e-04       3.016102e-03       2.869552e-02       5.306404e-03       1.235991e-02
```
We suggest that you plot steps (column 1) vs the maximum deviation of the forces (column 5), and the steps (column 1) vs the average deviation of the forces (column 7).
Are the value of the errors stable? What are their magnitudes? Can you conclude that the model is well-trained to describe the solid and liquid, or does it require further training?






Once the simulation has completed, analyze it with the same steps described above for the solid.

You can now copy the trajectory ```si.lammps-dump-text``` to your laptop and visualize it with Ovito.
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


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/04-Performing%20MD%20simulations/run_lammps.ipynb)
