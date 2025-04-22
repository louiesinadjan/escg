#!/bin/bash

# Number of parallel executions (default: 2 if not provided)
NUM_RUNS=${1:-2}

echo "Running ./build/metal $NUM_RUNS times in parallel..."

for ((i = 1; i <= NUM_RUNS; i++)); do
    time ./build/metal -m 10000 &  # Run in background
done

wait  # Wait for all parallel processes to finish

echo "All runs completed."