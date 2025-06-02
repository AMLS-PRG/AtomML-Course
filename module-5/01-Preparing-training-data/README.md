# Preparing raining data
——————————————————————————————————————————————————————

Note: Due to the high computational cost of MD simulations and DFT calculations—and since they are not the main focus of this tutorial—we have pre-generated both MD (in the section: 1. Exploration) and DFT (in the section: 2. Labeling) results for your convenience. The procedures used to obtain these results are described in detail within this document. In the final part of the tutorial, you will practice converting the DFT results into a dataset suitable for DeePMD-kit training (in the section: 2. Labeling).

——————————————————————————————————————————————————————
______________________________________________________

## Objectives

The objectives of this tutorial session are:
- Explore configurations for DFT calculations
- Prepare DFT output data for the training process


## 1. Exploration

Introduction: We will explore configurations for building a dataset, which consists of:
- Configurations of liquid silicon at 1700 K obtained in molecular dynamics simulations driven by the [Stillinger-Weber potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262).
- Configurations of silicon in the cubic diamond crystal structure obtained using random displacements from equilibrium atomic positions. 

*************************************Liquid Si - MD simulations with another force field*************************************

We will now run molecular dynamics simulation of liquid Si with the Stillinger-Weber force field using LAMMPS.
The LAMMPS input files can be found in the directory `module-5/01-Preparing-training-data/dataset/liquid-si-64/trajectory-lammps-1700K-1bar` for a simulation at 1 bar and 1700 K (approximate melting temperature of Stillinger-Weber Si).
The MD simulations can be run with the command,

```shell
lmp < start.lmp
```
and the simulation takes a couple of minutes to complete.
The atomic coordinates are written every 10 ps to the file `si.lammps-dump-text` in LAMMPS dump format. 

You can observe the LAMMPS trajectory. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Preparing-training-data/check_lammpsTrj.ipynb)


📌 You can:

- Inspect the structure and atomic configuration in each frame.
- Observe the **size of the simulation cell/model** in this trajectory.
- Later, **compare it with the model size** used in final MD simulations to better understand transferability and scaling.


> **Note** Element infomation can be saved to LAMMPS dump file if the followed commands are used. The `xs ys zs` are scaled coordinates. Other properties can be save as well, namely, atom velocities `vx vy vz`. (See more details in [doc](https://docs.lammps.org/dump.html).) When a dump file with element info is visualised by OVITO, particles will have corresponding radii and colours.
```
dump                    myDump all custom ${out_freq2} si.lammps-dump-text id type element xs ys zs
dump_modify             myDump element Si
```
> **Note** This trajectory does not reach the accuracy of first-principles calculations.


*************************************Crystalline Si - Random perturbations*************************************

Using this script (perturbations.py)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Preparing-training-data/perturbations.ipynb), first, the structurally optimized structure of bulk Si with 8 atoms will be read as ASE atoms object. 
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


## 2. Labeling

Introduction: Energies and forces for these configurations were obtained using DFT with the PBE functional.
You are encouraged to use the results of your own calculations.
However, you may also use the Quantum Espresso output files that we provide in the folders ```module-5/01-Preparing-training-data/dataset/liquid-si-64``` and ```module-5/01-Preparing-training-data/dataset/perturbations-si-64```.


*************************************Crystalline Si - Random perturbations*************************************

You have generated a set of atomic configurations from the exploration step (1. Exploration), the next step is to label these configurations, i.e., calculate energies and forces using DFT. The following `job.sh` bash script executes Quantum Espresso on the 100 input files that we just created by performing SCF DFT calculation for each frame to evaluate the forces and energy:

###### ***If you're already familiar with DFT calculations, feel free to skip this section.***
```shell
conda deactivate
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
for i in `seq 0 99`
do
        mpirun -np 1 $PW -input pw-si-$i.in > pw-si-$i.out
done
```
To run these DFT tasks in the background, you can use
```
chmod 777 job.sh
nohup ./job.sh &
```
To monitor the processes, you can use
```
ps aux|grep job.sh
ps aux|grep pw.x
```
If you want to shutdown the calculation, execute `kill PROCESSID` where `PROCESSID` is the id of the process `job.sh`.

###### ***If you're already familiar with DFT calculations, feel free to skip the section above.***

For each input file `pw-si-$i.in`, Quantum Espresso will create a `pw-si-$i.out` file which contains the potential energy, the forces, and other useful information. 


First, we have to extract the energies and forces from the Quantum Espresso output files and organize them in the .raw filetype suitable for DeePMD.
There are many ways to carry out this task.
Here, we propose to use a script ```get_raw.py```[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Preparing-training-data/get_raw.ipynb)
 based on [ASE](https://wiki.fysik.dtu.dk/ase/) that we provide in the folder ```module-5/01-Preparing-training-data/dataset/perturbations-si-64/0.2A-5p```.

You can parse the atomic structures, potential energy, and atomic forces in the the following files from QE outputs using the script ```get_raw.py``` [See the [manual](https://docs.deepmodeling.com/projects/deepmd/en/master/data/data-conv.html#raw-format-and-data-conversion) for an explanation of the format and units of these files.]:
- ```energy.raw```
- ```force.raw```
- ```coord.raw```
- ```box.raw```
- ```type.raw```

```python
import numpy as np
import ase.io
from ase.calculators.espresso import Espresso

# Open output files for writing
file_coord = open("coord.raw", "w")     # Coordinates
file_energy = open("energy.raw", "w")   # Potential energy
file_force = open("force.raw", "w")     # Forces
file_virial = open("virial.raw", "w")   # Virial stress
file_box = open("box.raw", "w")         # Cell dimensions
file_type = open("type.raw", "w")       # Atom types

types_written = False

for i in range(100):
    try:
        conf = ase.io.read('pw-si-' + str(i) + '.out', format='espresso-out')
    except:
        print("Configuration " + str(i) + " could not be read")
    else:
        try:
            conf.get_forces()
        except:
            print("Forces missing from file" + str(i))
        else:
            # Write data to respective output files
            file_coord.write(' '.join(conf.get_positions().flatten().astype('str').tolist()) + '\n')
            file_energy.write(str(conf.get_potential_energy()) + '\n')
            file_force.write(' '.join(conf.get_forces().flatten().astype('str').tolist()) + '\n')
            file_virial.write(' '.join(conf.get_stress(voigt=False).flatten().astype('str').tolist()) + '\n')
            file_box.write(' '.join(conf.get_cell().flatten().astype('str').tolist()) + '\n')
            
            if not types_written:
                types = np.array(conf.get_chemical_symbols())
                types[types == "Si"] = "0"
                file_type.write(' '.join(types.tolist()) + '\n')
                types_written = True

# Close output files
file_coord.close()
file_energy.close()
file_force.close()
file_virial.close()
file_box.close()
file_type.close()
```

Now let's verify if this script successfully generates the files `coord.raw`, `energy.raw`, `force.raw`, `virial.raw`, `box.raw`, and `type.raw`. It's important to note that while the raw format is not directly supported for training, NumPy and HDF5 binary formats are supported. So, in the next step, we have to extract the raw data from the PW outputs and convert them into the input format required by `deepMD-kit` for training (*.npy). A full list of these files can be found [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/data/system.md). The following is a description of the basic `deepMD-kit` input formats:

<br/>

<div align="center">
	
ID       | Property                | Raw file     | Shape                  
-------- | ----------------------  | ------------ | -----------------------
type     | Atom type indexes       | type.raw     | Natoms                 
coord    | Atomic coordinates      | coord.raw    | Nframes \* Natoms \* 3  in Å
box      | Boxes                   | box.raw      | Nframes \* 3 \* 3       in Å
energy   | Frame energies          | energy.raw   | Nframes                 in eV
force    | Atomic forces           | force.raw    | Nframes \* Natoms \* 3  in eV/Å
virial   | Frame virial            | virial.raw   | Nframes \* 9 in eV       

<em>The table is taken from [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/data/system.md). `Box` and `virial`: in the order `XX XY XZ YX YY YZ ZX ZY ZZ`.</em>
</div>

<br/>

To convert the prepared raw files to the NumPy, you can execute this utility in each folder containing .raw data files using the command:

```/your/path/deepmd-kit/data/raw/raw_to_set.sh 101```

The data should now be ready for the training process!

### !!! 
Another excellent way to directly convert output of electronic-structure calculation (pw-si-*.out) into the DeePMD-kit format (*.npy) is using [dpdata](https://docs.deepmodeling.com/projects/deepmd/en/master/data/dpdata.html). We also provide relevant examples to demonstrate this process.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Preparing-training-data/examples-for-dpdata/get_raw_files_dpdata.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Preparing-training-data/examples-for-dpdata/get_raw_files_ASE.ipynb)

### !!! 
All training datasets formatted for use with deepMD-kit (in .npy format) are available in module-5/01-Preparing-training-data/dataset/.


*************************************Liquid Si - MD simulations with another force field*************************************

We can now extract configurations from MD simulations trajectory and create input files to perform DFT calculations with the python script `get_configurations.py` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMLS-PRG/AtomML-Course/blob/main/module-5/01-Preparing-training-data/get_configurations.ipynb) which reads:
```python
import numpy as np
import ase.io
from ase.calculators.espresso import Espresso
import os

################################
# QE options
################################

pseudopotentials = {'Si': 'Si_ONCV_PBE-1.0.upf'}

input_qe = {
            'calculation':'scf',
            'outdir': './',             
            'pseudo_dir': './',         
            'tprnfor': True,        
            'tstress': True,        
            'disk_io':'none',
            'system':{
              'ecutwfc': 30,
              'input_dft': 'PBE',
              'occupations': 'smearing',
              'smearing': 'fermi-dirac',
              'degauss': 0.01,
             },
            'electrons':{
               'mixing_beta': 0.5,
               'electron_maxstep':1000,
             },
}

os.system('mkdir extracted-confs')

# Load trajectory
traj=ase.io.read('si.lammps-dump-text',format='lammps-dump-text',index=':')
step=1
counter1=0 # Number of configurations written
counter2=0 # Frame number
for conf in traj:
   if ((counter2%step)==0):
      species=np.array(conf.get_chemical_symbols())
      species=np.full(shape=species.shape,fill_value="Si")
      conf.set_chemical_symbols(species)
      ase.io.write('extracted-confs/pw-si-' + str(counter1) + '.in',conf, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials)
      counter1 += 1
   counter2 += 1
```
This will create a folder `extracted-confs` with 100 Quantum Espresso input files with the atomic configurations extracted from the trajectory `si.lammps-dump.text`.
You can now perform DFT calculations on all of these configurations using the same script `job.sh` as above.
Next, extract the raw data files with the script `get_raw.py` as above, and convert them into the appropriate format for `DeePMD-kit` using the `raw_to_set.sh` script.

The above calculations can be repeated for two other pressures, namely, +- 10 kbar, in order to sample a broad range of volumes.
This is illustrated in the folders `trajectory-lammps-1700K-10000bar` and `trajectory-lammps-1700K-neg10000bar`.

### !!! 
All training datasets formatted for use with deepMD-kit (in .npy format) are available in module-5/01-Preparing-training-data/dataset/.

> **Note** ASE can get the correct `chemical_symbols` if the LAMMPS dump file has `element` info. Otherwise, use `traj=ase.io.read('si.lammps-dump-text',format='lammps-dump-text',index=':',specorder=[“Si”])` to pass the chemical symbols. If there are two types in LAMMPS, namely, Si and O, then `specorder=["Si", "O"]`.









