#!/bin/bash

# This script installs the SARLens package and its dependencies

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Create a new Conda environment called AINavi
conda create -n SARLens python=3.10 -y

# Activate the AINavi environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate SARLens; then
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
python setup.py develop

# Define the filename for the configuration file
CONFIG_FILE="config.ini"

# Get the current directory path
CURRENT_DIR=$(pwd) || { echo "Error: Cannot get the current directory path." >&2; exit 1; }

# Write the configuration to the file
cat <<EOF > $CONFIG_FILE
[DATABASE]
DB_NAME = SARLens_db
DB_USER = myuser # This is the user that has access to the ASF database
DB_PASSWORD = mypassword # This is the password of the user

[LOGGING]
LOG_LEVEL = INFO

[DIRECTORIES]
SARLENS_DIR = $CURRENT_DIR
EOF

# Print a message indicating that the configuration file has been created
echo "Configuration file created at $(date) at $PWD."