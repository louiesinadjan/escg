#!/bin/bash

# Clear the results file before starting
echo "Starting benchmark trials..." > results.txt

# Run 10 IID trials for CURAND
echo "Running CURAND trials..." | tee -a results.txt
for i in {1..10}
do
    echo "Trial $i" | tee -a results.txt
    { time ./curand >> results.txt; } 2>> results.txt
    echo "------------------------------" | tee -a results.txt
done

# Run 10 IID trials for Mersenne Twister
echo "Running Mersenne Twister trials..." | tee -a results.txt
for i in {1..10}
do
    echo "Trial $i" | tee -a results.txt
    { time ./cuda-mersenne >> results.txt; } 2>> results.txt
    echo "------------------------------" | tee -a results.txt
done

echo "All trials completed." | tee -a results.txt