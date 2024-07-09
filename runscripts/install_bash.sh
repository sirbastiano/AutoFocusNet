#!/bin/bash

# This script installs the SARLens package and its dependencies

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Create a new Conda environment 
conda create -n sarlib python=3.9 -y

# Activate the AINavi environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate sarlib; then
    echo "SARLens environment activated"
else
    echo "SARLens environment not found"
    exit 1
fi

# Check if the script has been sourced
if [ -z "$BASH_SOURCE" ]; then
    echo "ERROR: You must source this script. Run 'source $0'"
    exit 1
fi
# Check if the script is being run from the right directory
if [ ! -f setup.py ]; then
  echo "ERROR: Run this from the top-level directory of the repository"
  exit 1
fi
# Run the setup.py script
python3 -m pip install --editable .

source setup.sh