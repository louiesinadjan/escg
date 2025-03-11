//
//  config.h
//  escg
//
//  Created by Louie Sinadjan on 06/01/2025.
//

#pragma once
#include <Foundation/Foundation.hpp> // For NS::String
#include <Metal/Metal.hpp>           // For MTL::Device, MTL::CommandQueue, MTL::Library, MTL::ComputePipelineState, MTL::Buffer
#include <string>                    // For std::string

// ------------------------------------------------------------------------------
// Structure to hold the input parameters of the simulation
// ------------------------------------------------------------------------------
struct Params {
    int MCS = 100000;           // Monte Carlo Steps
    int L = 200;                // Length of lattice
    int H = 200;                // Height of lattice
    int dimensions = 2;         // 1D, 2D, 3D
    int neighbourhood = 4;      // Von Neumann (4-way), Moore (8-way)
    int printFrequency = 200;   // MCS frequency to print snapshots
    float mobility = 1e-6;      // Mobility
    int species = 5;            // Number of species (Rock, Paper, Scissors, Lizard, Spock)
    bool flux = true;           // Flux boundary conditions
    float emptyProbability = 0; // Initial empty cell probability
    bool resume = false;        // Resume simulation from given files

    std::string outputDir; // Output directory for simulation files
};

//------------------------------------------------------------------------------
// Structure to hold Metal objects and configuration
//------------------------------------------------------------------------------
struct MetalContext {
    NS::AutoreleasePool* autoreleasePool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Library* library;
    MTL::ComputePipelineState* pipelineStateActions;    // Random action PSO
    MTL::ComputePipelineState* pipelineStateCells;      // Selected cell PSO
    MTL::ComputePipelineState* pipelineStateNeighbours; // Neighbour direction PSO
    MTL::Buffer* seedBuffer;                            // Seeds for random number generation
    MTL::Buffer* resultBufferActions;
    MTL::Buffer* resultBufferCells;
    MTL::Buffer* resultBufferNeighbours;

    MTL::ComputePipelineState* pipelineStateDensities; // Density PSO
    MTL::Buffer* gridBuffer;                           // GPU buffer for grid data
    MTL::Buffer* densityResultsBuffer;                 // GPU buffer for density results

    MTL::ComputePipelineState* pipelineStateStep; // Step PSO
    MTL::Buffer* cellsBuffer;                     // Cells to process
    MTL::Buffer* neighboursDirsBuffer;            // Neighbour directions to process
    MTL::Buffer* actionProbabilitiesBuffer;       // Actions to take
    MTL::Buffer* stepGridBuffer;                  // Grid

    int threads;
    int numRandomNumbers;
};

//------------------------------------------------------------------------------
// Structure to hold the grid data and visualisation vectors
//------------------------------------------------------------------------------
struct GridContext {
    int emptyCounter;
    int rockCounter;
    int paperCounter;
    int scissorsCounter;
    int lizardCounter;
    int spockCounter;

    std::vector<double> steps;
    std::vector<double> densityRock;
    std::vector<double> densityPaper;
    std::vector<double> densityScissors;
    std::vector<double> densityLizard;
    std::vector<double> densitySpock;
};

//------------------------------------------------------------------------------
// Structure to hold the step data to pass into the Metal shader
//------------------------------------------------------------------------------
struct StepContext {
    int* cells;
    int* neighbour_dirs;
    float* action_probabilities;
};

// Include CoreGraphics in Makefile or else the following error will occur:
//      Error: No Metal device available.
//      Failed to initialise Metal context.