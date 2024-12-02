#!/bin/bash
#SBATCH -p GPU                # partition (queue)
#SBATCH -N 1                  # number of nodes
#SBATCH -t 0-36:00            # time (D-HH:MM)
#SBATCH --gres=gpu:1          # Request 1 GPU

# Default values for arguments
OUTPUT_DIRS=""
OLLAMA_LOG_DIR=""
RUN_NUMBERS=""

# Parse options
while getopts ":o:l:n:" opt; do
  case $opt in
    o)
      OUTPUT_DIRS=$OPTARG
      ;;
    l)
      OLLAMA_LOG_DIR=$OPTARG
      ;;
    n)
      RUN_NUMBERS=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if the output directory and job number are provided
if [ -z "$OUTPUT_DIRS" ] || [ -z "$OLLAMA_LOG_DIR" ] || [ -z "$RUN_NUMBERS" ]; then
  echo "output directories (-o) and ollama log directory (-l) and run numbers (-n) must be provided."
  exit 1
fi

CONDA_PATH="/usr/local/anaconda3"
ENV_NAME="hyde"
OLLAMA_LOG="${OLLAMA_LOG_DIR}/ollama${RUN_NUMBERS}.log"
PYTHON_SCRIPT="./src/experiment2.py"

# Load the appropriate environment
if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    . "${CONDA_PATH}/etc/profile.d/conda.sh"
else
    export PATH="${CONDA_PATH}/bin:$PATH"
fi

# Activate the conda environment
source activate ${ENV_NAME}

# Set correct tmp dirs
TMPDIR=/home/u440541/tmp 
TEMP=/home/u440541/tmp 
OLLAMA_TMPDIR=/tmp 
OLLAMA_MODELS=/tmp/ollama

# Start the Ollama server in the background and log the output
echo "Starting Ollama server and pulling model..."
ollama serve >> ${OLLAMA_LOG} 2>&1 &
# Wait a bit to ensure the server starts
sleep 10
ollama pull llama3.1 >> ${OLLAMA_LOG} 2>&1 &

for OUTPUT_DIR in $OUTPUT_DIRS; do
  # Run the Python script in the background with the output directory as an argument
  echo "Running the Python script with output directory ${OUTPUT_DIR}..."
  python ${PYTHON_SCRIPT} --run_directory ${OUTPUT_DIR} &
done

# Wait for all background jobs to finish
wait

# Print a message after all jobs have completed
echo "All jobs completed."
