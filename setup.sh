#!/bin/bash

# Define the source directory where contriever_msmarco_index should be installed
SRC_DIR="./src"
INDEX_DIR="$SRC_DIR/contriever_msmarco_index"

# Check if the contriever_msmarco_index directory exists in the ./src directory
if [ ! -d "$INDEX_DIR" ]; then
  echo "contriever_msmarco_index directory not found in ./src. Downloading and extracting..."

  # Create the src directory if it doesn't exist
  mkdir -p $SRC_DIR

  # Download contriever_msmarco_index.tar.gz
  wget https://www.dropbox.com/s/dytqaqngaupp884/contriever_msmarco_index.tar.gz

  # Extract the tar.gz file to the ./src directory
  echo "Extracting contriever_msmarco_index.tar.gz to ./src..."
  tar -xvf contriever_msmarco_index.tar.gz -C $SRC_DIR
else
  echo "contriever_msmarco_index directory already exists in ./src. Skipping download."
fi

# Name of the conda environment
ENV_NAME="hyde"

# Add conda-forge channel if it's not already added
conda config --add channels conda-forge
conda config --set channel_priority strict

# Check if the conda environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with Python 3.9 and OpenJDK 21..."
    conda create -y -n $ENV_NAME python=3.9 openjdk=21
fi

# Activate the conda environment
echo "Activating conda environment '$ENV_NAME'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing required Python modules from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please ensure it is in the current directory."
fi

echo "Environment setup complete."
