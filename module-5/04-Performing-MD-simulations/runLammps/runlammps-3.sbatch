#!/bin/bash
#SBATCH --partition=pintxo
#SBATCH --account=pintxo
#SBATCH --qos=pintxo
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G         # memory per cpu-core (4G is default)
#SBATCH --time=10-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="test_solid"
#SBATCH --gres=gpu:1             # number of gpus per node

pwd; hostname; date

module purge
module load Python DeePMD/2.2.10-gpu CUDA

#mv out out-02
#mkdir out
#cp /scratch/zhouwq/DFT-feldspar-surface-small/MD-110-01/in-singleP.deepmd .
#cp /scratch/zhouwq/DFT-feldspar-surface-small/MD-110-01/in-singleP.setup .
#cp /scratch/zhouwq/DFT-feldspar-surface-small/MD-110-01/start.lmp .
#cp /scratch/zhouwq/DFT-feldspar-surface-small/MD-110-01/in.dump .
#mv md.out md-add02.out
#lmp -in start-npt-270K.lmp
lmp -in input_solid_local.lmp

date
