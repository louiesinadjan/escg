#!/bin/bash

num_trials=100
alpha=0 
beta=0 
skip_trial=false
trap 'echo -e "\nSkipping current trial..."; skip_trial=true' SIGINT

for ((trial=1; trial<=num_trials; trial++)); do
    echo "Running simulation with alpha=$alpha, beta=$beta, trial=$trial"
    ./escg -a $alpha -b $beta -l 100
done

echo "All simulations completed."