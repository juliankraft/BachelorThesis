#!/usr/bin/env bash

#SBATCH --job-name=jk-BA
#SBATCH --mail-type=fail,end
#SBATCH --time=01-00:00:00
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
cd /cfs/earth/scratch/kraftjul/BA/hpc_submit/calculating_featurestats
pwd
echo
echo '#########################################################################################'
echo
echo '### Updating Packages ###################################################################'
source /cfs/earth/scratch/kraftjul/BA_package/update_stable.sh
echo
echo '#########################################################################################'
echo '### Running script ######################################################################'
echo '#########################################################################################'
echo
micromamba run -n mega python /cfs/earth/scratch/kraftjul/BA/code/run/calculating_featurestats.py
echo
echo '#########################################################################################'
echo '### Completed script ####################################################################'
echo '#########################################################################################'