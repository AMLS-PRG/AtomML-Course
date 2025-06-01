## Training data

______________________________________________________
**1. Exploration**

We will use the training data for silicon prepared on [hand-on session 3](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1/3-preparing-training-data).
These data consist of:
- Around 400 configurations of silicon in the cubic diamond crystal structure obtained using random displacements from equilibrium atomic positions. 
- Around 300 configurations of liquid silicon at 1700 K obtained in molecular dynamics simulations driven by the [Stillinger-Weber potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262).

#Details for training data
Crystalline Si - Random perturbations

Using this script, first, the structurally optimized structure of bulk Si with 8 atoms will be read as ASE atoms object. 
Then the supercell will be constructed by expanding the unit cell using (2 x 2 x 2) transformation vector which yields the supercell with 64 atoms.

```python
from ase.build import make_supercell

bulk_si = ase.io.read('../pw-si-vc_relax.out',format='espresso-out')
P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
conf = make_supercell(bulk_si, P)
```

Then, you will apply random displacements to the atomic positions of a bulk Si supercell and vary the lattice parameters using ASE atoms object. Random displacements within the defined maximum displacement will be added to the equilibrium atomic positions, and random fractional changes within the defined maximum cell change will be applied to the lattice parameters. The Python script will generate a total of `100 frames` and corresponding QE input files. In each frame, the cell and the atomic positions will be perturbed by a maximum of `1 %` and `0.01 Å`, respectively, from the ground state bulk Si structure. The degree of perturbation of the atomic positions and the cell for each frame follows a uniform distribution.

```python
initial_positions = supercell.get_positions()
initial_cell = supercell.get_cell()

max_displacement=0.01 # Maximum displacement in angstrom
max_cell_change=0.01  # Maximum fractional change in cell

num_iterations=100

for i in range(num_iterations):
    positions=np.copy(initial_positions)
    cell=np.copy(initial_cell)

    # Displace each coordinate randomly
    positions += np.random.rand(positions.shape[0],positions.shape[1])*2*max_displacement - max_displacement
    conf.set_positions(positions)

    # Scale each cell component randomly
    cell *= 1-(np.random.rand(cell.shape[0],cell.shape[1])*2*max_cell_change-max_cell_change)
    conf.set_cell(cell,scale_atoms=True)
	
    # Write QE input
    write('pw-si-' + str(i) + '.in',conf, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials)
```

Let's type `python perturbations.py` to generate QE input files. Let's play with the `max_displacement` and `max_cell_change` variables by constructing the different datasets to sample enough chemical spaces (refer to `0.05A-2p`, `0.1A-3p`, `0.2A-5p` directories).

<br/>

Liquid Si - MD simulations with another force field

**1. Exploration**: We will now run molecular dynamics simulation of liquid Si with the Stillinger-Weber force field using LAMMPS.
The LAMMPS input files can be found in the directory `liquid-si-64/trajectory-lammps-1700K-1bar` for a simulation at 1 bar and 1700 K (approximate melting temperature of Stillinger-Weber Si).
The MD simulations can be run with the command,

```shell
lmp < start.lmp
```
and the simulation takes a couple of minutes to complete.
The atomic coordinates are written every 10 ps to the file `si.lammps-dump-text` in LAMMPS dump format.

> **Note** Element infomation can be saved to LAMMPS dump file if the followed commands are used. The `xs ys zs` are scaled coordinates. Other properties can be save as well, namely, atom velocities `vx vy vz`. (See more details in [doc](https://docs.lammps.org/dump.html).) When a dump file with element info is visualised by OVITO, particles will have corresponding radii and colours.
```
dump                    myDump all custom ${out_freq2} si.lammps-dump-text id type element xs ys zs
dump_modify             myDump element Si
```






Energies and forces for these configurations were obtained using DFT with the PBE functional.
You are encouraged to use the results of your own calculations.
However, you may also use the Quantum Espresso output files that we provide in the folders ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/example-data/liquid-si-64``` and ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/example-data/perturbations-si-64```.

First, we have to extract the energies and forces from the Quantum Espresso output files and organize them in the .raw filetype suitable for DeePMD.
There are many ways to carry out this task.
Here, we propose to use a script ```get_raw.py``` based on [ASE](https://wiki.fysik.dtu.dk/ase/) that we provide in the folder ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/scripts/```.
You can execute this script in the folders containing the Quantum Espresso output files to obtain the following files:
- ```energy.raw```
- ```force.raw```
- ```coord.raw```
- ```box.raw```
- ```type.raw```

See the [manual](https://docs.deepmodeling.com/projects/deepmd/en/master/data/data-conv.html#raw-format-and-data-conversion) for an explanation of the format and units of these files.
The last step is to use the ```raw_to_set.sh``` utility in DeePMD to have the data ready for the training process.
You can execute this utility in each folder containing .raw data files using the command:

```/home/deepmd23admin/Softwares/deepmd-kit/data/raw/raw_to_set.sh 101```

The data should now be ready for the training process!
Another excellent way to convert output of electronic-structure calculation into the DeePMD-kit format is using [dpdata](https://docs.deepmodeling.com/projects/deepmd/en/master/data/dpdata.html).

We shall see below whether a DeePMD model trained on the configurations described above is able to drive the dynamics of this system, while preserving a high-accuracy.
Can you guess if the model will be good for the liquid, the solid, none, or both? Why?
Let's make a poll in the classroom!
