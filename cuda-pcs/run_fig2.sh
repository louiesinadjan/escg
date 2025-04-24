#!/bin/bash

step=0.05
num_trials=20

skip_trial=false
trap 'echo -e "\nSkipping current trial..."; skip_trial=true' SIGINT

for ((trial=1; trial<=num_trials; trial++)); do
    for alpha in $(seq 0 $step 1); do
        for beta in $(seq 0 $step 1); do
            if $skip_trial; then
                skip_trial=false 
                continue
            fi

            echo "Running simulation with alpha=$alpha, beta=$beta, trial=$trial"
            ./cuda-escg -a $alpha -b $beta -g 1 -l 400 -m 160000 -x true --numRandoms 40000000 -o pcs_fig2.csv
        done
    done
done

echo "All simulations completed."