#!/bin/bash

# Default values
NUM_RUNS=8
RUNS_DIR="./runs/runs_2"
START_SCRIPT="./start_script.sh"

# Create the run directory if it doesn't exist
if [ ! -d "$RUNS_DIR" ]; then
    echo "Creating directory $RUNS_DIR"
    mkdir -p "$RUNS_DIR"
fi

# Function to submit a job
submit_job() {
  local run_number=$1
  # Set output and error file paths
  run_dirs=""
  run_numbers=""
  for ((i=run_number; i<=run_number+2; i++)); do
    local run_dir="$RUNS_DIR/run_$i"
    # Create a directory for the current run
    mkdir -p "$run_dir"
    run_dirs+="$run_dir "
    run_numbers+="$i"
  done
  echo "$run_numbers"
  local output_file="$RUNS_DIR/slurm${run_numbers}.out"
  local error_file="$RUNS_DIR/slurm${run_numbers}.err"

  echo "$output_file"
  
  # If no server was provided, try to find an idle server
  SERVER=$(sinfo -o "%N %T" | awk '$2 == "idle" {print $1}' | tr ',' '\n' | head -n 1)
  while [ -z "$SERVER" ]; do
    echo "No idle server found. Checking again in 10 seconds."
    sleep 10
    SERVER=$(sinfo -o "%N %T" | awk '$2 == "idle" {print $1}' | tr ',' '\n' | head -n 1)
  done
  echo "Found idle server: $SERVER"
  
  # Submit the job with the appropriate output, error files, job number, and output directory
  echo "Submitting run $run_number to server $SERVER"
  sbatch --nodelist="$SERVER" -o "$output_file" -e "$error_file" "$START_SCRIPT" -o "$run_dirs" -l $RUNS_DIR -n $run_numbers
  # sbatch -o "$output_file" -e "$error_file" "$START_SCRIPT" -o "$run_dirs" -l $RUNS_DIR -n $run_numbers
}

# Run the script 10 times
for ((i=1; i<=NUM_RUNS*3; i+=3)); do
  submit_job $i
  echo "Run $i submitted, waiting 10 seconds before checking for the next run."
  sleep 10
done
