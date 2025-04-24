#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <vector>  

struct Params {
    int MCS = 100000;           // Monte Carlo Steps
    int L = 200;                // Length of lattice
    int H = 200;                // Height of lattice
    int neighbourhood = 4;      // Von Neumann (4-way), Moore (8-way)
    int printFrequency = 200;   // MCS frequency to print snapshots
    bool patchwork = false;
    int species = 8;            // Number of species (Rock, Paper, Scissors, Lizard, Spock)
    bool flux = true;           // Flux boundary conditions
    float emptyProbability = 0; // Initial empty cell probability
    int numRandoms = 40'000'000;  // Number of random numbers to generate

    bool save = false;      // Save grid and snapshots?
    bool dominance = false; // Import dominance.csv?
    bool resume = false;    // Resume simulation from given files?

        // Params for Park Chen Szolnoki
    float alpha = 0.5;
    float beta = 0.5;
    float gamma = 1;
    
    bool maxStep = false; // Process numRandoms number of MCS per metal call 

    std::string resultsFile; // File to save results
    std::string outputDir = "results.csv"; // Output directory for simulation files
};

struct GridContext {
    std::vector<double> steps;
    std::vector<std::vector<int>> speciesCounters;
    std::vector<std::vector<double>> speciesDensities;
};

struct StepContext {
    int* cells;
    int* neighbour_dirs;
    float* action_probabilities;
};