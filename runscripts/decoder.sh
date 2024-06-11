#!/bin/bash
# redirect stdout/stderr to a file
# remove logfile if existing:
rm -f logfile_decoding.log
exec >logfile_decoding.log 2>&1


clear
# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Activate the SARLens environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate SARLens; then
    echo "SARLens environment activated"
else
    echo "SARLens environment not found"
    exit 1
fi


Now1=$(date +"%Y-%m-%d %H:%M:%S")
echo "========= Start Decoding ========= $Now1"
echo "Input product is:" $1
echo "Output directory is:" $2
# python -m processor.decode --inputfile $1 --output $2

python -m processor.decode --inputfile $1 --output /Users/robertodelprete/Desktop/AutoFocusNet/DATA/DECODED
Now2=$(date +"%Y-%m-%d %H:%M:%S")
echo "========= End Decoding ========= $Now2"


exit 0
