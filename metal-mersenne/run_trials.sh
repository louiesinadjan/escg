#!/bin/bash

# Clear the results file before starting
echo "Starting benchmark trials..." > results.txt

# Run 10 IID trials for CURAND
echo "Running Serial Mersenne trials..." | tee -a results.txt
for i in {1..10}
do
    echo "Trial $i" | tee -a results.txt
    { time ./build/serial-mersenne >> results.txt; } 2>> results.txt
    echo "------------------------------" | tee -a results.txt
done

# Run 10 IID trials for Cuda-Mersenne Twister
echo "Running Metal Mersenne trials..." | tee -a results.txt
for i in {1..10}
do
    echo "Trial $i" | tee -a results.txt
    { time ./build/metal-mersenne >> results.txt; } 2>> results.txt
    echo "------------------------------" | tee -a results.txt
done

echo "All trials completed." | tee -a results.txt