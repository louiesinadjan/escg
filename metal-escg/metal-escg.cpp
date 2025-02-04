#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.hpp"
#include "simulation.hpp"
#include "timer.hpp"
#include "visualise.hpp"
#include <chrono>
#include <iostream>
#include <random>

//------------------------------------------------------------------------------
// Define a context structure to hold Metal objects and configuration
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
    MTL::Buffer* densityBuffer;                        // GPU buffer for density results

    int threads;
    int numRandomNumbers;
};

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
// Initialisation: Create device, load library, compile pipelines, create buffers
//------------------------------------------------------------------------------
bool initMetalContext(MetalContext& ctx) {
    ctx.numRandomNumbers = 100000000; // 100,000,000  random numbers per shader
    ctx.threads = ctx.numRandomNumbers / 10000;

    ctx.autoreleasePool = NS::AutoreleasePool::alloc()->init();

    // Create Metal device and command queue
    ctx.device = MTL::CreateSystemDefaultDevice();
    if (!ctx.device) {
        std::cerr << "Error: No Metal device available." << std::endl;
        return false;
    }
    ctx.commandQueue = ctx.device->newCommandQueue();

    // Load the Metal library
    NS::Error* error = nullptr;
    NS::String* libraryPath = NS::String::string("/Users/louiesinadjan/Documents/dissertation/escg/metal-escg/build/escg.metallib", NS::UTF8StringEncoding);

    ctx.library = ctx.device->newLibrary(libraryPath, &error);
    if (!ctx.library) {
        std::cerr << "Error: Failed to load escg.metallib - " << error->localizedDescription()->utf8String() << std::endl;
        return false;
    }

    // Load the shader functions from the library
    // Randoms
    MTL::Function* random_actions = ctx.library->newFunction(NS::String::string("mt_random_actions", NS::UTF8StringEncoding));
    MTL::Function* random_cells = ctx.library->newFunction(NS::String::string("mt_random_cells", NS::UTF8StringEncoding));
    MTL::Function* random_neighbours = ctx.library->newFunction(NS::String::string("mt_random_neighbours", NS::UTF8StringEncoding));

    // Densities
    MTL::Function* computeDensities = ctx.library->newFunction(NS::String::string("compute_densities", NS::UTF8StringEncoding)); //

    if (!random_actions || !random_cells || !random_neighbours || !computeDensities) {
        std::cerr << "Error: Failed to find one or more functions in Metal library." << std::endl;
        return false;
    }

    // Create compute pipeline states
    // Randoms
    ctx.pipelineStateActions = ctx.device->newComputePipelineState(random_actions, &error);
    ctx.pipelineStateCells = ctx.device->newComputePipelineState(random_cells, &error);
    ctx.pipelineStateNeighbours = ctx.device->newComputePipelineState(random_neighbours, &error);

    // Densities
    ctx.pipelineStateDensities = ctx.device->newComputePipelineState(computeDensities, &error);

    if (!ctx.pipelineStateActions || !ctx.pipelineStateCells || !ctx.pipelineStateNeighbours || !ctx.pipelineStateDensities) {
        std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        return false;
    }

    // Prepare the seed buffer
    uint32_t* seeds = new uint32_t[ctx.threads];
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);
    for (int i = 0; i < ctx.threads; i++) {
        seeds[i] = i; // dist(gen);
    }
    ctx.seedBuffer = ctx.device->newBuffer(seeds, sizeof(uint32_t) * ctx.threads, MTL::ResourceStorageModeShared);
    delete[] seeds;

    // Create result buffers
    // Randoms
    ctx.resultBufferActions = ctx.device->newBuffer(sizeof(float) * ctx.numRandomNumbers, MTL::ResourceStorageModeShared);
    ctx.resultBufferCells = ctx.device->newBuffer(sizeof(uint32_t) * ctx.numRandomNumbers, MTL::ResourceStorageModeShared);
    ctx.resultBufferNeighbours = ctx.device->newBuffer(sizeof(uint32_t) * ctx.numRandomNumbers, MTL::ResourceStorageModeShared);

    // Densities
    ctx.gridBuffer = ctx.device->newBuffer(sizeof(int) * 40000, MTL::ResourceStorageModeShared);
    ctx.densityBuffer = ctx.device->newBuffer(sizeof(int) * 6, MTL::ResourceStorageModeShared); // 6 species counts (RPSLS + E)

    return true;
}

//------------------------------------------------------------------------------
// Refresh: Use existing pipeline and buffer objects to generate new random numbers
//------------------------------------------------------------------------------
void refreshRandomNumbers(MetalContext& ctx, float* action_probabilities, uint32_t* cells, uint32_t* neighbours) {
    std::cout << "Refreshing random numbers" << std::endl;

    // Setup common dispatch parameters
    MTL::Size gridSize = MTL::Size(ctx.threads, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(ctx.pipelineStateActions->maxTotalThreadsPerThreadgroup(), 1, 1);

    // 1. Refresh random actions
    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(ctx.pipelineStateActions);
    encoder->setBuffer(ctx.seedBuffer, 0, 0);
    encoder->setBuffer(ctx.resultBufferActions, 0, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    std::memcpy(action_probabilities, ctx.resultBufferActions->contents(), sizeof(float) * ctx.numRandomNumbers);

    // 2. Refresh random cells
    commandBuffer = ctx.commandQueue->commandBuffer();
    encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(ctx.pipelineStateCells);
    encoder->setBuffer(ctx.seedBuffer, 0, 0);
    encoder->setBuffer(ctx.resultBufferCells, 0, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    std::memcpy(cells, ctx.resultBufferCells->contents(), sizeof(uint32_t) * ctx.numRandomNumbers);

    // 3. Refresh random neighbours
    commandBuffer = ctx.commandQueue->commandBuffer();
    encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(ctx.pipelineStateNeighbours);
    encoder->setBuffer(ctx.seedBuffer, 0, 0);
    encoder->setBuffer(ctx.resultBufferNeighbours, 0, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    std::memcpy(neighbours, ctx.resultBufferNeighbours->contents(), sizeof(uint32_t) * ctx.numRandomNumbers);
}

//------------------------------------------------------------------------------
// Cleanup: Release all allocated Metal objects
//------------------------------------------------------------------------------
void destroyMetalContext(MetalContext& ctx) {
    if (ctx.seedBuffer)
        ctx.seedBuffer->release();
    if (ctx.resultBufferActions)
        ctx.resultBufferActions->release();
    if (ctx.resultBufferCells)
        ctx.resultBufferCells->release();
    if (ctx.resultBufferNeighbours)
        ctx.resultBufferNeighbours->release();
    if (ctx.pipelineStateActions)
        ctx.pipelineStateActions->release();
    if (ctx.pipelineStateCells)
        ctx.pipelineStateCells->release();
    if (ctx.pipelineStateNeighbours)
        ctx.pipelineStateNeighbours->release();
    if (ctx.library)
        ctx.library->release();
    if (ctx.commandQueue)
        ctx.commandQueue->release();
    if (ctx.device)
        ctx.device->release();
    if (ctx.autoreleasePool)
        ctx.autoreleasePool->release();
}

//------------------------------------------------------------------------------
// Calculate densities of each species and add to vectors for visualisation using metal
//------------------------------------------------------------------------------
void computeDensitiesGPU(MetalContext& ctx, int grid[200][200], int* densities) {
    // Flatten the grid into a 1D array
    int flattenedGrid[40000]; // 200x200 → 1D array
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            flattenedGrid[i * 200 + j] = grid[i][j]; // Flatten row-wise
        }
    }

    // Copy the flattened grid to the GPU buffer
    std::memcpy(ctx.gridBuffer->contents(), flattenedGrid, sizeof(int) * 40000);
    // Reset the density buffer
    std::memset(ctx.densityBuffer->contents(), 0, sizeof(int) * 6);

    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(ctx.pipelineStateDensities);
    encoder->setBuffer(ctx.gridBuffer, 0, 0);
    encoder->setBuffer(ctx.densityBuffer, 0, 1); // Set buffer for atomic operations

    MTL::Size threadsPerGrid = MTL::Size(40000, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(200, 1, 1); // Adjust for GPU

    encoder->dispatchThreads(threadsPerGrid, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Copy the results from the GPU buffer to the densities array
    std::memcpy(densities, ctx.densityBuffer->contents(), sizeof(int) * 6);
}

//------------------------------------------------------------------------------
// Calculate densities of each species and add to vectors for visualisation
//------------------------------------------------------------------------------
void densities(int grid[200][200], int L, int mcs, GridContext& gridCtx, MetalContext& metalCtx) {
    int speciesCounts[6] = {0}; // [empty, rock, paper, scissors, lizard, spock]

    // Call Metal shader
    computeDensitiesGPU(metalCtx, grid, speciesCounts);

    int totalCells = L * L;

    double emptyDensity = (static_cast<double>(speciesCounts[0]) / totalCells) * 100;
    double rockDensity = (static_cast<double>(speciesCounts[1]) / totalCells) * 100;
    double paperDensity = (static_cast<double>(speciesCounts[2]) / totalCells) * 100;
    double scissorsDensity = (static_cast<double>(speciesCounts[3]) / totalCells) * 100;
    double lizardDensity = (static_cast<double>(speciesCounts[4]) / totalCells) * 100;
    double spockDensity = (static_cast<double>(speciesCounts[5]) / totalCells) * 100;

    gridCtx.steps.push_back(mcs);
    gridCtx.densityRock.push_back(rockDensity);
    gridCtx.densityPaper.push_back(paperDensity);
    gridCtx.densityScissors.push_back(scissorsDensity);
    gridCtx.densityLizard.push_back(lizardDensity);
    gridCtx.densitySpock.push_back(spockDensity);

    // Print the densities
    if (mcs % 200 == 0) {
        std::cout << "Population Densities at: " << mcs << std::endl;
        std::cout << "EMPTY: " << emptyDensity << std::endl;
        std::cout << "ROCK: " << rockDensity << std::endl;
        std::cout << "PAPER: " << paperDensity << std::endl;
        std::cout << "SCISSORS: " << scissorsDensity << std::endl;
        std::cout << "LIZARD: " << lizardDensity << std::endl;
        std::cout << "SPOCK: " << spockDensity << std::endl;
        std::cout << std::endl;
    }
}

//------------------------------------------------------------------------------
// Plot density against steps
//------------------------------------------------------------------------------
void show(GridContext& gridCtx) { plot_densities(gridCtx.steps, gridCtx.densityRock, gridCtx.densityPaper, gridCtx.densityScissors, gridCtx.densityLizard, gridCtx.densitySpock); }

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    // ------------------- Timer -------------------

    Timer timer;
    timer.start();

    // ------------------- Metal Parameters -------------------

    const int numRandomNumbers = 100000000;
    float* action_probabilities = new float[numRandomNumbers];
    uint32_t* cells = new uint32_t[numRandomNumbers];
    uint32_t* neighbours = new uint32_t[numRandomNumbers];
    int index = 0;

    MetalContext metalCtx;
    if (!initMetalContext(metalCtx)) {
        std::cerr << "Failed to initialise Metal context." << std::endl;
        return -1;
    }

    refreshRandomNumbers(metalCtx, action_probabilities, cells, neighbours);

    // ------------------- Simulation Parameters -------------------

    int MCS = 100000;
    GridContext gridCtx;

    int L = 200;        // Length of lattice
    int N = L * L;      // Elementary time steps
    float M = 1e-6f;    // Mobility 'since it is proportional to the typical area
                        // explored by one mobile individual per unit time'
    int grid[200][200]; // Grid size = 200 x 200

    float mu = 1;              // RPSLS interaction
    float sigma = 1;           // Reproduction
    float epsilon = 2 * M * N; // Migration

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 5); // Range: 1 to 5 (RPSLS)

    // Randomly initialise the grid with species (stored as int)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i][j] = dist(gen); // Randomly assign a species (or EMPTY) as an integer
        }
    }

    // ------------------- Start Simulating -------------------

    for (int mcs = 0; mcs <= MCS; mcs++) { // Monte Carlo Steps
        if (index >= numRandomNumbers) {
            refreshRandomNumbers(metalCtx, action_probabilities, cells, neighbours); // Fill random numbers
            index = 0;                                                               // Reset index after refreshing random numbers
        }

        densities(grid, L, mcs, gridCtx, metalCtx); // Every MCS, call densities to add to density vectors for visualisation after simulation
        if (mcs == 0 || mcs == 2000 || mcs == 6000 || mcs == 20000 || mcs == 100000) {
            plot_snapshot(grid, L, mcs);
        }

        // if (mcs % 10 == 0) {
        //     plot_snapshot(grid, L, mcs);
        // }

        for (int n = 0; n < N; n++) { // Elementary Time Steps

            int cell = cells[index];                         // Random number 0-39999
            int neighbour = neighbours[index];               // Random number 0-3
            float action_prob = action_probabilities[index]; // Random number 0-1
            index++;

            // static std::random_device rd;
            // static std::mt19937 gen(rd());                               // Random number generator
            // std::uniform_int_distribution<int> dist_pos(0, (L * L) - 1); // Random position in grid
            // std::uniform_int_distribution<int> dist_dir(0, 3);           // Random neighbour direction (0 to 3 for four neighbours)
            // std::uniform_real_distribution<float> dist_prob(0.0, 1);     // Random probability for actions

            // cell = dist_pos(gen);         // Random cell
            // neighbour = dist_dir(gen);    // Random neighbour
            // action_prob = dist_prob(gen); // Random action probability

            // Normalise mu + sigma + epsilon
            float sum = mu + sigma + epsilon;
            mu /= sum;
            sigma /= sum;
            epsilon /= sum;

            int action;
            if (action_prob < mu) {
                action = 1; // Interaction
            } else if (action_prob < mu + sigma) {
                action = 2; // Reproduction
            } else {
                action = 3; // Migration
            }

            step(L, grid, cell, neighbour, action);
        }
    }

    show(); // Plot density against steps

    std::cout << "Simulation Complete.";

    destroyMetalContext(metalCtx);
    delete[] action_probabilities;
    delete[] cells;
    delete[] neighbours;

    // ------------------- End Timer -------------------

    timer.stop();
    std::cout << "Time taken: " << timer.elapsedSeconds() << "s" << std::endl;

    return 0;
}
