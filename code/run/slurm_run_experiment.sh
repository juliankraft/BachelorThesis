#!/usr/bin/env bash

#SBATCH --job-name=jk_BA
#SBATCH --mail-type=fail,end
#SBATCH --time=04-00:00:00
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
ENV_NAME="mega"
PYTHON_SCRIPT="/cfs/earth/scratch/kraftjul/BA_package/ba_stable/run_experiment.py"
CONFIG_PATH="/cfs/earth/scratch/kraftjul/BA/hpc_submit/run_experiment/experiment_config.yaml"

echo '#########################################################################################'
echo '### Setup: ##############################################################################'
echo '#########################################################################################'
echo
echo "Running on host: $(hostname)"
echo
nvidia-smi
echo
echo "Using env:"
echo " ${ENV_NAME}"
echo "Script:"
echo " ${PYTHON_SCRIPT}"
echo "Config:"
echo " ${CONFIG_PATH}"
echo
echo '#########################################################################################'
echo '### Running script ######################################################################'
echo '#########################################################################################'
echo
micromamba run -n "${ENV_NAME}" python "${PYTHON_SCRIPT}" \
    --config_path "${CONFIG_PATH}"
echo
echo '#########################################################################################'
echo '### Completed script ####################################################################'
echo '#########################################################################################'