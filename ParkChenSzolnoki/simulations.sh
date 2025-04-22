#!/bin/bash

# Define the step size
step=0.05

# Define the number of trials per alpha-beta pair
num_trials=10

# Function to handle SIGINT (Ctrl+C)
skip_trial=false
trap 'echo -e "\nSkipping current trial..."; skip_trial=true' SIGINT

# Run the simulations for alpha and beta from 0 to 1 in steps of 0.05
for ((trial=1; trial<=num_trials; trial++)); do
    for alpha in $(seq 0 $step 1); do
        for beta in $(seq 0 $step 1); do
            if $skip_trial; then
                skip_trial=false  # Reset flag for the next trial
                continue
            fi

            echo "Running simulation with alpha=$alpha, beta=$beta, trial=$trial"
            ./build/metal -a $alpha -b $beta -l 100 -m 10000
        done
    done
done

echo "All simulations completed."