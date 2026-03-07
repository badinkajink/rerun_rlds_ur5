#!/bin/bash

echo "Creating Conda environment with Python 3.12"
conda create -n deligrasp_env python=3.12 -y

echo "Initializing Conda for this script..."

eval "$(conda shell.bash hook)"

echo "Activating environment..."
conda activate deligrasp_env

echo "Installing dm-tree via conda-forge..."
conda install -c conda-forge dm-tree -y

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Environment setup complete!"