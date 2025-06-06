#!/bin/bash
#SBATCH --qos=regular
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=1-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="si_solid-test"
#SBATCH --gres=gpu:1             # number of gpus per node

pwd; hostname; date

module purge
module load Python DeePMD/2.2.10-gpu CUDA

lmp -in input_solid_local.lmp


date
