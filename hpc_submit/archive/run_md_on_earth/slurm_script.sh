#!/usr/bin/env bash

#SBATCH --job-name=test_md
#SBATCH --mail-type=fail,end
#SBATCH --time=02-00:00:00
#SBATCH --qos=earth-4.4d
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=42G
#SBATCH --gres=gpu:l40s:1

# shellcheck disable=SC1091

# ## strict bash mode
set -eEuo pipefail

# module reset
module purge
module load DefaultModules

# ## init micromamba
export MAMBA_ROOT_PREFIX="/cfs/earth/scratch/${USER}/.conda/"
eval "$("/cfs/earth/scratch/${USER}/bin/micromamba" shell hook -s posix)"

echo '#########################################################################################'
echo '### Host info: ##########################################################################'
echo
echo 'Running on host:'
hostname
echo
nvidia-smi
echo
echo 'Working directory:'
cd /cfs/earth/scratch/kraftjul/BA/hpc_submit/run_md_on_earth
pwd
echo
echo '#########################################################################################'
echo
echo '#########################################################################################'
echo '### Running skript ######################################################################'
echo '#########################################################################################'
echo

micromamba run -n mega python run_md_on_earth.py
echo
echo '#########################################################################################'
echo '### Completed skript ####################################################################'
echo '#########################################################################################'