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

void metal(float*& action_probabilities, uint32_t*& cells, uint32_t*& neighbours) {
    std::cout << "Refreshing random numbers." << std::endl;

    NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Load the Metal library from random.metallib
    NS::Error* error = nullptr;
    NS::String* libraryPath = NS::String::string("/Users/louiesinadjan/Documents/dissertation/escg/metal-escg/build/random.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libraryPath, &error);
    if (!library) {
        std::cerr << "Error: Failed to load random.metallib - " << error->localizedDescription()->utf8String() << std::endl;
        autoreleasePool->release();
        return;
    }

    std::cout << "Metal Library Loaded" << std::endl;

    // Load the functions from the Metal library
    MTL::Function* random_actions = library->newFunction(NS::String::string("mt_random_actions", NS::UTF8StringEncoding));
    MTL::Function* random_cells = library->newFunction(NS::String::string("mt_random_cells", NS::UTF8StringEncoding));
    MTL::Function* random_neighbours = library->newFunction(NS::String::string("mt_random_neighbours", NS::UTF8StringEncoding));

    if (!random_actions || !random_cells || !random_neighbours) {
        std::cerr << "Error: Failed to find one or more functions in random.metallib." << std::endl;
        autoreleasePool->release();
        return;
    }

    // Create compute pipeline states
    MTL::ComputePipelineState* pipelineStateActions = device->newComputePipelineState(random_actions, &error);
    MTL::ComputePipelineState* pipelineStateCells = device->newComputePipelineState(random_cells, &error);
    MTL::ComputePipelineState* pipelineStateNeighbours = device->newComputePipelineState(random_neighbours, &error);

    if (!pipelineStateActions || !pipelineStateCells || !pipelineStateNeighbours) {
        std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        autoreleasePool->release();
        return;
    }

    // Prepare buffers
    const int threads = 10000; // Number of threads that work in parallel in metal shader
    const int numRandomNumbers = 100000000;
    uint32_t* seeds = new uint32_t[threads];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);
    for (int i = 0; i < threads; ++i) {
        seeds[i] = dist(gen);
    }

    MTL::Buffer* seedBuffer = device->newBuffer(seeds, sizeof(uint32_t) * threads, MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBufferActions = device->newBuffer(sizeof(float) * numRandomNumbers, MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBufferCells = device->newBuffer(sizeof(uint32_t) * numRandomNumbers, MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBufferNeighbours = device->newBuffer(sizeof(uint32_t) * numRandomNumbers, MTL::ResourceStorageModeShared);

    // Create command buffer and encoder for random actions
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineStateActions);
    encoder->setBuffer(seedBuffer, 0, 0);
    encoder->setBuffer(resultBufferActions, 0, 1);

    // Dispatch threads
    MTL::Size gridSize = MTL::Size(threads, 1, 1); // Set gridSize to the number of threads
    MTL::Size threadGroupSize = MTL::Size(pipelineStateActions->maxTotalThreadsPerThreadgroup(), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();

    // Commit command buffer and wait for completion
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Retrieve results
    std::memcpy(action_probabilities, resultBufferActions->contents(), sizeof(float) * numRandomNumbers);

    // Repeat for random cells
    commandBuffer = commandQueue->commandBuffer();
    encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineStateCells);
    encoder->setBuffer(seedBuffer, 0, 0);
    encoder->setBuffer(resultBufferCells, 0, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    std::memcpy(cells, resultBufferCells->contents(), sizeof(uint32_t) * numRandomNumbers);

    // Repeat for random neighbours
    commandBuffer = commandQueue->commandBuffer();
    encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineStateNeighbours);
    encoder->setBuffer(seedBuffer, 0, 0);
    encoder->setBuffer(resultBufferNeighbours, 0, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    std::memcpy(neighbours, resultBufferNeighbours->contents(), sizeof(uint32_t) * numRandomNumbers);

    delete[] seeds;
    autoreleasePool->release();
}

int main(int argc, const char* argv[]) {
    // ------------------- Metal Parameters -------------------

    const int numRandomNumbers = 100000000;
    float* action_probabilities = new float[numRandomNumbers];
    uint32_t* cells = new uint32_t[numRandomNumbers];
    uint32_t* neighbours = new uint32_t[numRandomNumbers];
    int index = 0;
    metal(action_probabilities, cells, neighbours);

    // ------------------- Simulation Parameters -------------------

    int MCS = 2000;

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

    int zero = 0, one = 0, two = 0, three = 0;
    int lastZero = 0, lastOne = 0, lastTwo = 0, lastThree = 0;
    for (int i = 0; i <= 100000; i++) {
        if (i % 40 == 0) {
            std::cout << "Counted: " << i << std::endl;
            std::cout << "0: " << zero - lastZero << " \t\tTotal: " << zero << std::endl;
            std::cout << "1: " << one - lastOne << ", \t\tTotal: " << one << std::endl;
            std::cout << "2: " << two - lastTwo << ", \t\tTotal: " << two << std::endl;
            std::cout << "3: " << three - lastThree << ", \t\tTotal: " << three << std::endl;

            lastZero = zero;
            lastOne = one;
            lastTwo = two;
            lastThree = three;
        }
        if (neighbours[i] == 0) {
            zero++;
        } else if (neighbours[i] == 1) {
            one++;
        } else if (neighbours[i] == 2) {
            two++;
        } else if (neighbours[i] == 3) {
            three++;
        } else {
            std::cout << "Invalid neighbour: " << neighbours[i] << std::endl;
        }
    }

    // for (int i = 0; i < numRandomNumbers; i++) {
    //     std::cout << i << ": " << action_probabilities[i] << ",\t" << cells[i] << ",\t" << neighbours[i] << std::endl;
    // }

    // Before starting simulation, calculate average of action_probabilities
    // double sum = 0.0;
    // for (int i = 0; i < numRandomNumbers; i++) {
    //     sum += action_probabilities[i];
    //     if(action_probabilities[i] < 0 || action_probabilities[i] > 1) {
    //         std::cout << "Invalid action_probabilities: " << action_probabilities[i] << std::endl;
    //         return -1;
    //     }
    // }
    // double average = sum / numRandomNumbers;
    // std::cout << "Average of action_probabilities: " << average << std::endl;
    // std::cout << "Expected average should be close to 0.5" << std::endl;

    // sum = 0;
    // for (int i = 0; i < numRandomNumbers; i++) {
    //     sum += cells[i];
    //     if(cells[i] < 0 || cells[i] > 39999) {
    //         std::cout << "Invalid cells: " << cells[i] << std::endl;
    //         return -1;
    //     }
    // }
    // average = sum / numRandomNumbers;
    // std::cout << "Average of cells: " << average << std::endl;

    // sum = 0;
    // for (int i = 0; i < numRandomNumbers; i++) {
    //     sum += neighbours[i];
    //     if(neighbours[i] < 0 || neighbours[i] > 3) {
    //         std::cout << "Invalid neighbours: " << neighbours[i] << std::endl;
    //         return -1;
    //     }
    // }
    // average = sum / numRandomNumbers;
    // std::cout << "Average of neighbours: " << average << std::endl;

    // ------------------- Start Simulating -------------------

    // for (int mcs = 0; mcs <= MCS; mcs++) { // Monte Carlo Steps
    //     if (index >= numRandomNumbers) {
    //         metal(action_probabilities, cells, neighbours); // Refresh random numbers
    //         index = 0;                                      // Reset index after refreshing random numbers
    //     }

    //     densities(grid, L, mcs); // Every MCS, call densities to add to density vectors for visualisation after simulation
    //     // if (mcs == 0 || mcs == 2000 || mcs == 6000 || mcs == 20000 || mcs == 100000) {
    //     //     plot_snapshot(grid, L, mcs);
    //     // }

    //     if (mcs % 10 == 0) {
    //         plot_snapshot(grid, L, mcs);
    //     }

    //     for (int n = 0; n < N; n++) { // Elementary Time Steps
    //         int cell = cells[index];
    //         int neighbour = neighbours[index];
    //         float action_prob = action_probabilities[index];
    //         index++;

    //         // normalise mu + sigma + epsilon
    //         float sum = mu + sigma + epsilon;
    //         mu /= sum;
    //         sigma /= sum;
    //         epsilon /= sum;

    //         int action;
    //         if (action_prob < mu) {
    //             action = 1; // Interaction
    //         } else if (action_prob < mu + sigma) {
    //             action = 2; // Reproduction
    //         } else {
    //             action = 3; // Migration
    //         }

    //         step(L, grid, cell, neighbour, action);
    //     }
    // }

    // show(); // Plot density against steps

    // std::cout << "Simulation Complete.";
    // delete[] action_probabilities;
    // delete[] cells;
    // delete[] neighbours;

    return 0;
}
