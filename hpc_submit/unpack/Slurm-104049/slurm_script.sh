#!/usr/bin/env bash

#SBATCH --job-name=unpack
#SBATCH --mail-type=FAIL,END
#SBATCH --time=02:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Increased CPU for faster extraction
#SBATCH --mem=32G  # Increased memory for large dataset

# Define source and target directories
ZIP_FILE="/cfs/earth/scratch/kraftjul/BA/data_zip/dataset.zip"
TARGET_DIR="/cfs/earth/scratch/kraftjul/BA/"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Unpack the dataset efficiently
unzip -q "$ZIP_FILE" -d "$TARGET_DIR"
