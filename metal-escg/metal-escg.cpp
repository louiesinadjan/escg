#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.hpp"
#include "io.hpp"
#include "visualise.hpp"
#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

//------------------------------------------------------------------------------
// Parse command line arguments
//------------------------------------------------------------------------------
Params parseArgs(int argc, char* argv[]) {
    Params params; // Uses default values

    static struct option long_options[] = {
        {"mcs", required_argument, 0, 'm'},           {"length", required_argument, 0, 'l'},
        {"height", required_argument, 0, 'h'},        {"printFrequency", required_argument, 0, 'p'},
        {"neighbourhood", required_argument, 0, 'n'}, {"species", required_argument, 0, 's'},
        {"mobility", required_argument, 0, 'M'},      {"flux", required_argument, 0, 'f'},
        {"empty", required_argument, 0, 'w'},         {0, 0, 0, 0} // End of options
    };

    int opt;
    int option_index = 0;

    std::string usage = " [--mcs <MCS>] [--length <Lattice Length>] [--height <Lattice Height>] "
                        " [--printFrequency <Print Frequency>] [--empty <Initial Empty Cell Probability >] "
                        "[--neighbourhood <Neighbourhood 4/8>] [--mobility <Mobility>] "
                        "[--species <Number of Species 3/5>] [--flux <true|false>]"
                        " [--resume <true|false>]";

    // Parse the command line arguments
    while ((opt = getopt_long(argc, argv, "m:l:h:p:n:M:s:f:e:r:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                params.MCS = std::stoi(optarg);
                break;
            case 'l':
                params.L = std::stoi(optarg);
                break;
            case 'h':
                params.H = std::stoi(optarg);
                break;
            case 'p':
                params.printFrequency = std::stoi(optarg);
                break;
            case 'n':
                params.neighbourhood = std::stoi(optarg);
                if (params.neighbourhood != 4 && params.neighbourhood != 8) {
                    std::cerr << "Error: Neighbourhood must be 4 or 8." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 's':
                params.species = std::stoi(optarg);
                if (params.species != 3 && params.species != 5) {
                    std::cerr << "Error: Number of species must be 3 or 5." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'M':
                params.mobility = std::stof(optarg); // Allows scientific notation input
                break;
            case 'f': {
                std::string fluxStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : fluxStr) {
                    c = tolower(c);
                }
                if (fluxStr == "false" || fluxStr == "0" || fluxStr == "no") {
                    params.flux = false;
                } else {
                    params.flux = true;
                }
                break;
            }
            case 'e':
                params.emptyProbability = std::stof(optarg);
                if (params.emptyProbability <= 0 || params.emptyProbability > 1) {
                    std::cerr << "Error: Initial empty cell probability must be 0 <= p < 1." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'r': {
                std::string resumeStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : resumeStr) {
                    c = tolower(c);
                }
                if (resumeStr == "true" || resumeStr == "1" || resumeStr == "yes") {
                    params.resume = true;
                } else {
                    params.resume = false;
                }
                break;
            }
            default:
                std::cerr << "Usage: " << argv[0] << usage << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    return params;
}
//------------------------------------------------------------------------------
// Initialisation: Create device, load library, compile pipelines, create buffers
//------------------------------------------------------------------------------
bool initMetalContext(MetalContext& ctx, int N, int randomNums) {
    ctx.numRandomNumbers = randomNums;          // 100,000,000  random numbers per shader
    ctx.threads = ctx.numRandomNumbers / 10000; // 10,000 numbers per thread

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
    MTL::Function* computeDensities = ctx.library->newFunction(NS::String::string("compute_densities", NS::UTF8StringEncoding));

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
    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000);
    uint32_t* seeds = new uint32_t[ctx.threads];
    for (int i = 0; i < ctx.threads; i++) {
        // seeds[i] = i;
        dist(gen);
    }
    ctx.seedBuffer = ctx.device->newBuffer(seeds, sizeof(uint32_t) * ctx.threads, MTL::ResourceStorageModeShared);
    delete[] seeds;

    // Create result buffers
    // Randoms
    ctx.resultBufferActions = ctx.device->newBuffer(sizeof(float) * ctx.numRandomNumbers, MTL::ResourceStorageModeShared);
    ctx.resultBufferCells = ctx.device->newBuffer(sizeof(uint32_t) * ctx.numRandomNumbers, MTL::ResourceStorageModeShared);
    ctx.resultBufferNeighbours = ctx.device->newBuffer(sizeof(uint32_t) * ctx.numRandomNumbers, MTL::ResourceStorageModeShared);

    // Densities
    ctx.gridBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);           // Buffer size of L * L
    ctx.densityResultsBuffer = ctx.device->newBuffer(sizeof(int) * 6, MTL::ResourceStorageModeShared); // 6 species counts (RPSLS + E)

    return true;
}

//------------------------------------------------------------------------------
// Refresh: Use existing pipeline and buffer objects to generate new random numbers
//------------------------------------------------------------------------------
void refreshRandomNumbers(MetalContext& ctx, float* action_probabilities, uint32_t* cells, uint32_t* neighbours, int N, bool moore) {
    std::cout << "Refreshing random numbers\n" << std::endl;

    // Setup common dispatch parameters
    MTL::Size gridSize = MTL::Size(ctx.threads, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(ctx.pipelineStateActions->maxTotalThreadsPerThreadgroup(), 1, 1);

    // Refresh random actions buffers
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

    // Refresh random cells buffers
    commandBuffer = ctx.commandQueue->commandBuffer();
    encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(ctx.pipelineStateCells);
    encoder->setBuffer(ctx.seedBuffer, 0, 0);
    encoder->setBuffer(ctx.resultBufferCells, 0, 1);
    encoder->setBytes(&N, sizeof(int), 2); // Size of grid, pick a random number from 0 to N - 1
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    std::memcpy(cells, ctx.resultBufferCells->contents(), sizeof(uint32_t) * ctx.numRandomNumbers);

    // Refresh random neighbours buffers
    commandBuffer = ctx.commandQueue->commandBuffer();
    encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(ctx.pipelineStateNeighbours);
    encoder->setBuffer(ctx.seedBuffer, 0, 0);
    encoder->setBuffer(ctx.resultBufferNeighbours, 0, 1);
    encoder->setBytes(&moore, sizeof(bool), 2); // Moore or Von Neumann neighbourhood
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
    if (ctx.pipelineStateDensities)
        ctx.pipelineStateDensities->release();
    if (ctx.gridBuffer)
        ctx.gridBuffer->release();
    if (ctx.densityResultsBuffer)
        ctx.densityResultsBuffer->release();
    if (ctx.pipelineStateStep)
        ctx.pipelineStateStep->release();
    if (ctx.cellsBuffer)
        ctx.cellsBuffer->release();
    if (ctx.neighboursDirsBuffer)
        ctx.neighboursDirsBuffer->release();
    if (ctx.actionProbabilitiesBuffer)
        ctx.actionProbabilitiesBuffer->release();
    if (ctx.stepGridBuffer)
        ctx.stepGridBuffer->release();
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
void computeDensitiesGPU(MetalContext& ctx, int* grid, int* densities, int N) {
    // Copy the flattened grid to the GPU buffer
    std::memcpy(ctx.gridBuffer->contents(), grid, sizeof(int) * N);
    // Reset the density buffer
    std::memset(ctx.densityResultsBuffer->contents(), 0, sizeof(int) * 6);

    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(ctx.pipelineStateDensities);
    encoder->setBuffer(ctx.gridBuffer, 0, 0);
    encoder->setBuffer(ctx.densityResultsBuffer, 0, 1); // Set buffer for atomic operations

    MTL::Size threadsPerGrid = MTL::Size(N, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(ctx.pipelineStateDensities->maxTotalThreadsPerThreadgroup(), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Copy the results from the GPU buffer to the densities array
    std::memcpy(densities, ctx.densityResultsBuffer->contents(), sizeof(int) * 6);
}

//------------------------------------------------------------------------------
// Calculate densities of each species and add to vectors for visualisation
//------------------------------------------------------------------------------
void densities(int* grid, int N, int mcs, GridContext& gridCtx, MetalContext& metalCtx, int printFrequency) {
    int speciesCounts[6] = {0}; // [empty, rock, paper, scissors, lizard, spock]

    // Call Metal shader
    computeDensitiesGPU(metalCtx, grid, speciesCounts, N);

    // Calculate the percentage density cells
    double emptyDensity = (static_cast<double>(speciesCounts[0]) / N) * 100;
    double rockDensity = (static_cast<double>(speciesCounts[1]) / N) * 100;
    double paperDensity = (static_cast<double>(speciesCounts[2]) / N) * 100;
    double scissorsDensity = (static_cast<double>(speciesCounts[3]) / N) * 100;
    double lizardDensity = (static_cast<double>(speciesCounts[4]) / N) * 100;
    double spockDensity = (static_cast<double>(speciesCounts[5]) / N) * 100;

    gridCtx.steps.push_back(mcs);
    gridCtx.densityRock.push_back(rockDensity);
    gridCtx.densityPaper.push_back(paperDensity);
    gridCtx.densityScissors.push_back(scissorsDensity);
    gridCtx.densityLizard.push_back(lizardDensity);
    gridCtx.densitySpock.push_back(spockDensity);

    // Print the densities
    if (mcs % printFrequency == 0 && mcs != 0) {
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
void show(GridContext& gridCtx, Params params) { plot_densities(gridCtx, params); }

//------------------------------------------------------------------------------
// Initialise metal step buffers and pipelines
// Computes the step functionality
//------------------------------------------------------------------------------
void initMetalStep(MetalContext& ctx, int N) {
    NS::Error* error = nullptr;

    // Load the step function from the library
    MTL::Function* stepFunction = ctx.library->newFunction(NS::String::string("step", NS::UTF8StringEncoding));
    if (!stepFunction) {
        std::cerr << "Error: Failed to find step function in Metal library." << std::endl;
        return;
    }

    ctx.pipelineStateStep = ctx.device->newComputePipelineState(stepFunction, &error);
    if (!ctx.pipelineStateStep) {
        std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        return;
    }

    ctx.cellsBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);
    ctx.neighboursDirsBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);
    ctx.actionProbabilitiesBuffer = ctx.device->newBuffer(sizeof(float) * N, MTL::ResourceStorageModeShared);
    ctx.stepGridBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);
}

// ------------------------------------------------------------------------------
// Metal step shader
// ------------------------------------------------------------------------------
void metalStep(MetalContext& ctx, StepContext& stepCtx, float mu, float sigma, int N, Params& p, int* grid) {
    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    // Assign to the buffers
    std::memcpy(ctx.cellsBuffer->contents(), stepCtx.cells, sizeof(int) * N);
    std::memcpy(ctx.neighboursDirsBuffer->contents(), stepCtx.neighbour_dirs, sizeof(int) * N);
    std::memcpy(ctx.actionProbabilitiesBuffer->contents(), stepCtx.action_probabilities, sizeof(float) * N);
    std::memcpy(ctx.stepGridBuffer->contents(), grid, sizeof(int) * N);

    // Create buffers for cells, neighbour directions, action probabilities
    encoder->setComputePipelineState(ctx.pipelineStateStep);
    encoder->setBuffer(ctx.cellsBuffer, 0, 0);
    encoder->setBuffer(ctx.neighboursDirsBuffer, 0, 1);
    encoder->setBuffer(ctx.actionProbabilitiesBuffer, 0, 2);

    // Using `setBytes()` for scalar values (floats)
    encoder->setBytes(&mu, sizeof(float), 3);
    encoder->setBytes(&sigma, sizeof(float), 4);
    encoder->setBytes(&p.L, sizeof(int), 5);
    encoder->setBytes(&p.H, sizeof(int), 6);
    encoder->setBytes(&p.flux, sizeof(bool), 7);

    // Set the grid buffer
    encoder->setBuffer(ctx.stepGridBuffer, 0, 8);

    MTL::Size threadsPerGrid = MTL::Size(1000, 1, 1); // 40,000 cells, 1,000 threads --> 40 cells per thread
    MTL::Size threadGroupSize = MTL::Size(ctx.pipelineStateStep->maxTotalThreadsPerThreadgroup(), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void initialiseGrid(int* grid, Params p) {
    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, p.species);     // Range: 1 to 5 (RPSLS)
    std::uniform_real_distribution<float> emptyCellProb(0, 1); // Range: 0 to 1

    // Randomly initialise the grid with species (stored as int)
    for (int i = 0; i < p.L * p.H; i++) {
        if (emptyCellProb(gen) < p.emptyProbability) {
            grid[i] = 0; // Randomly assign an empty cell as an integer
        } else {
            grid[i] = dist(gen); // Randomly assign a species (or EMPTY) as an integer
        }
    }
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    // Debug prints to check argc and argv

    // ------------------- Parse Command Line Arguments -------------------
    Params params = parseArgs(argc, const_cast<char**>(argv));

    int currentMCS = 0;
    int MCS = params.MCS;
    int L;      // Length of lattice
    int H;      // Height of lattice
    int N;      // Elementary time steps = total number of cells
    bool moore; // Moore neighbourhood if true, Von Neumann if false
    int* grid;  // Flattened grid size = L x H

    if (params.resume) {
        std::cout << "Resuming simulation from previous state." << std::endl;
        importCSVToParams(params); // Doesn't rewrite target MCS

        // MCS = params.MCS;
        L = params.L;
        H = params.H;
        N = L * H;
        moore = params.neighbourhood == 8;
        grid = new int[N];

        currentMCS = importCSVToGrid(grid, N); // Assigns to grid and returns the current MCS

        // Creating output directory
        std::string length = "l" + std::to_string(L);
        std::string height = "h" + std::to_string(H);
        std::string neighbourhood = moore ? "Moore" : "VN";
        std::ostringstream oss;
        oss.precision(2);
        oss << std::scientific << params.mobility;
        std::string mobility_str = "M" + oss.str();
        std::string flux = params.flux ? "flux" : "noflux";
        std::string species = std::to_string(params.species) + "species";
        params.outputDir = length + "_" + height + "_" + neighbourhood + "_" + mobility_str + "_" + flux + "_" + species;
    } else {
        std::cout << "Starting new simulation.\n" << std::endl;

        L = params.L;
        H = params.H;
        N = L * H;
        MCS = params.MCS;
        moore = params.neighbourhood == 8;
        grid = new int[N];

        // Creating output directory
        std::string length = "l" + std::to_string(L);
        std::string height = "h" + std::to_string(H);
        std::string neighbourhood = moore ? "Moore" : "VN";
        std::ostringstream oss;
        oss.precision(2);
        oss << std::scientific << params.mobility;
        std::string mobility_str = "M" + oss.str();
        std::string flux = params.flux ? "flux" : "noflux";
        std::string species = std::to_string(params.species) + "species";
        params.outputDir = length + "_" + height + "_" + neighbourhood + "_" + mobility_str + "_" + flux + "_" + species;
    }

    // Create the directory
    if (std::filesystem::create_directory(params.outputDir)) {
        std::cout << "Directory created: " << params.outputDir << std::endl;
    } else {
        std::cout << "Failed to create or already exists: " << params.outputDir << std::endl;
    }

    exportParamsToCSV(params);                 // Export the parameters to a csv file
    exportGridToCSV(grid, params, currentMCS); // Export the grid to a csv file

    std::cout << "------------------- Parameters -------------------\n";
    std::cout << "MCS: " << params.MCS << "\n";
    std::cout << "Lattice Length: " << params.L << "\n";
    std::cout << "Lattice Height: " << params.H << "\n";
    std::cout << "Initial Empty Cell Probability: " << params.emptyProbability << "\n";
    std::cout << "Neighbourhood: " << params.neighbourhood << "\n";
    std::cout << "Mobility: " << params.mobility << "\n";
    std::cout << "Species: " << params.species << "\n";
    std::cout << "Flux: " << params.flux << "\n";
    std::cout << "Print Frequency: " << params.printFrequency << "\n";
    std::cout << "Resume: " << params.resume << "\n";
    std::cout << "-------------------------------------------------\n";

    // ------------------- Metal Parameters -------------------

    const int numRandomNumbers = 100000000;
    float* action_probabilities = new float[numRandomNumbers];
    uint32_t* cells = new uint32_t[numRandomNumbers];
    uint32_t* neighbours = new uint32_t[numRandomNumbers];
    int index = 0;

    MetalContext metalCtx;
    if (!initMetalContext(metalCtx, N, numRandomNumbers)) {
        std::cerr << "Failed to initialise Metal context." << std::endl;
        return -1;
    }

    refreshRandomNumbers(metalCtx, action_probabilities, cells, neighbours, N, moore);

    // ------------------- Simulation Parameters -------------------

    GridContext gridCtx;
    StepContext stepCtx;

    stepCtx.cells = new int[N];
    stepCtx.neighbour_dirs = new int[N];
    stepCtx.action_probabilities = new float[N];

    initMetalStep(metalCtx, N); // Initialise Metal step buffers and pipelines

    float M = params.mobility; // Mobility 'since it is proportional to the typical area
                               // explored by one mobile individual per unit time'

    float mu = 1;              // RPSLS interaction
    float sigma = 1;           // Reproduction
    float epsilon = 2 * M * N; // Migration

    // Normalise the action probabilities
    float sum = mu + sigma + epsilon;
    mu /= sum;
    sigma /= sum;
    epsilon /= sum;

    if (params.resume) {
        std::cout << "Resuming simulation from MCS: " << currentMCS << std::endl;
    } else {
        initialiseGrid(grid, params); // Initialise the grid
    }

    // ------------------- Start Simulating -------------------

    plot_snapshot(grid, currentMCS, params);              // Plot snapshots at specific MCS
    exportGridToCSV(grid, params, currentMCS);            // Export the grid to a csv file
    densities(grid, N, currentMCS, gridCtx, metalCtx, 1); // Print initial densities

    for (int mcs = currentMCS; mcs <= MCS; mcs++) {                        // Monte Carlos
        densities(grid, N, mcs, gridCtx, metalCtx, params.printFrequency); // Every MCS, call densities to add to density vectors for visualisation after simulation
        if (mcs == 2000 || mcs == 6000 || mcs == 20000 || mcs == 100000) {
            plot_snapshot(grid, mcs, params);   // Plot snapshots at specific MCS
            exportGridToCSV(grid, params, mcs); // Export the grid to a csv file
        }

        // Fill the arrays with the next N cells, neighbour directions, and action probabilities
        for (int i = 0; i < N; i++) {
            if (index >= numRandomNumbers) {
                refreshRandomNumbers(metalCtx, action_probabilities, cells, neighbours, N, moore); // Fill random numbers
                index = 0;                                                                         // Reset index after refreshing random numbers
            }

            stepCtx.cells[i] = cells[index];
            stepCtx.neighbour_dirs[i] = neighbours[index];
            stepCtx.action_probabilities[i] = action_probabilities[index];

            index++;
        }

        std::memcpy(metalCtx.stepGridBuffer->contents(), grid, sizeof(int) * N);
        metalStep(metalCtx, stepCtx, mu, sigma, N, params, grid);
        std::memcpy(grid, metalCtx.stepGridBuffer->contents(), sizeof(int) * N);
    }

    show(gridCtx, params); // Plot density against steps

    std::cout << "Simulation Complete.";

    destroyMetalContext(metalCtx);
    delete[] action_probabilities;
    delete[] cells;
    delete[] neighbours;

    return 0;
}
