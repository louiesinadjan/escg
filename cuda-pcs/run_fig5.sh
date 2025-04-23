#!/bin/bash

step=0.05
num_trials=10

skip_trial=false
trap 'echo -e "\nSkipping current trial..."; skip_trial=true' SIGINT

for ((trial=1; trial<=num_trials; trial++)); do
    for alpha in $(seq 0 $step 0.4); do
        if $skip_trial; then
            skip_trial=false 
            continue
        fi

        echo "Running simulation with alpha=$alpha, trial=$trial"
        ./cuda-escg -a $alpha --beta 0.75 --gamma 1 -l 100 -m 500000 -x true --numRandoms 40000000
    done
done

for ((trial=1; trial<=num_trials; trial++)); do
    for alpha in $(seq 0 $step 0.4); do
        if $skip_trial; then
            skip_trial=false 
            continue
        fi

        echo "Running simulation with alpha=$alpha, trial=$trial"
        ./cuda-escg -a $alpha --beta 0.75 --gamma 1 -l 800 -m 20000 -x true --numRandoms 32000000
    done
done

echo "All simulations completed."