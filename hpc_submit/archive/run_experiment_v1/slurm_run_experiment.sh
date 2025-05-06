#!/usr/bin/env bash

#SBATCH --job-name=jk_BA
#SBATCH --mail-type=fail,end
#SBATCH --time=00-12:00:00
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

# ## setup
CONFIG_PATH="/cfs/earth/scratch/kraftjul/BA/hpc_submit/run_experiment/experiment_config.yaml"
LOG_DIR="/cfs/earth/scratch/kraftjul/BA/output/runs/efficient_v2"
rm -rf ${LOG_DIR} || true
mkdir -p ${LOG_DIR}

echo '#########################################################################################'
echo '### Host info: ##########################################################################'
echo
echo 'Running on host:'
hostname
echo
nvidia-smi
echo
echo 'Working directory:'
cd /cfs/earth/scratch/kraftjul/BA/code/run
pwd
echo
echo
echo '#########################################################################################'
echo '### Running script ######################################################################'
echo '#########################################################################################'
echo
micromamba run -n mega python run_experiment.py \
    --config_path "${CONFIG_PATH}" \
    --log_dir "${LOG_DIR}"

echo
echo '#########################################################################################'
echo '### Completed script ####################################################################'
echo '#########################################################################################'