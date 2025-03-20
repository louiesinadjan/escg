#!/bin/bash

# Serial execution benchmark
echo "Running 2 trials serially..."
time ( ./build/metal -a 0.5 -b 0.5 -l 100 
       ./build/metal -a 0.5 -b 0.5 -l 100 )

echo "Running 2 trials in parallel..."
# Parallel execution benchmark
time ( ./build/metal -a 0.5 -b 0.5 -l 100 &
       ./build/metal -a 0.5 -b 0.5 -l 100 &
       wait )

exit 0
