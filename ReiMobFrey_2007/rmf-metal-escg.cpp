#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.hpp"
#include "visualise.hpp"
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <random>
#include <vector>

struct Params {
    int MCS = 100000;         // Monte Carlo Steps
    int L = 200;              // Length of lattice
    int H = 200;              // Height of lattice
    int dimensions = 2;       // 1D, 2D, 3D
    int neighbourhood = 4;    // Von Neumann (4-way), Moore (8-way)
    int printFrequency = 200; // MCS frequency to print snapshots
    float mobility = 3e-6;    // Mobility
};

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

struct StepContext {
    int* cells;
    int* neighbour_dirs;
    float* action_probabilities;
};

Params parseArgs(int argc, char* argv[]) {
    Params params; // Uses default values

    static struct option long_options[] = {
        {"mcs", required_argument, 0, 'm'},
        {"length", required_argument, 0, 'l'},
        {"height", required_argument, 0, 'h'},
        {"dimensions", required_argument, 0, 'd'},
        {"printFrequency", required_argument, 0, 'p'},
        {"neighbourhood", required_argument, 0, 'n'},
        {"mobility", required_argument, 0, 'M'},
        {0, 0, 0, 0} // End of options
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "m:l:h:d:p:n:M:", long_options, &option_index)) != -1) {
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
            case 'd':
                params.dimensions = std::stoi(optarg);
                break;
            case 'p':
                params.printFrequency = std::stoi(optarg);
                break;
            case 'n':
                params.neighbourhood = std::stoi(optarg);
                break;
            case 'M':
                params.mobility = std::stof(optarg); // **Allows scientific notation input**
                break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [--mcs <MCS>] [--length <Lattice Length>] [--height <Lattice Height>] "
                             "[--dimensions <Dimensions>] [--printFrequency <Print Frequency>] "
                             "[--neighbourhood <Neighbourhood>] [--mobility <Mobility>]\n";
                exit(EXIT_FAILURE);
        }
    }

    return params;
}

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

    // Needs the full path to the .metallib file
    NS::String* libraryPath = NS::String::string("/Users/louiesinadjan/Documents/dissertation/escg/ReiMobFrey_2007/build/escg.metallib", NS::UTF8StringEncoding);

    ctx.library = ctx.device->newLibrary(libraryPath, &error);
    if (!ctx.library) {
        std::cerr << "Error: Failed to load escg.metallib - " << error->localizedDescription()->utf8String() << std::endl;
        return false;
    }

    // Load the shader functions from the library
    // Randoms
    MTL::Function* random_actions = ctx.library->newFunction(NS::String::string("rmf_mt_random_actions", NS::UTF8StringEncoding));
    MTL::Function* random_cells = ctx.library->newFunction(NS::String::string("rmf_mt_random_cells", NS::UTF8StringEncoding));
    MTL::Function* random_neighbours = ctx.library->newFunction(NS::String::string("rmf_mt_random_neighbours", NS::UTF8StringEncoding));

    // Densities
    MTL::Function* computeDensities = ctx.library->newFunction(NS::String::string("rmf_compute_densities", NS::UTF8StringEncoding));

    if (!random_actions || !random_cells || !random_neighbours || !computeDensities) {
        std::cerr << "Error: Failed to find one or more functions in Metal library." << std::endl;
        std::cout << random_actions << std::endl;
        std::cout << random_cells << std::endl;
        std::cout << random_neighbours << std::endl;
        std::cout << computeDensities << std::endl;
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

void refreshRandomNumbers(MetalContext& ctx, float* action_probabilities, uint32_t* cells, uint32_t* neighbours, int N, bool moore) {
    std::cout << "Refreshing random numbers" << std::endl;

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
    if (mcs % printFrequency == 0) {
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

void show(GridContext& gridCtx) { plot_densities(gridCtx.steps, gridCtx.densityRock, gridCtx.densityPaper, gridCtx.densityScissors, gridCtx.densityLizard, gridCtx.densitySpock); }

void initMetalStep(MetalContext& ctx, StepContext& stepCtx, int N) {
    NS::Error* error = nullptr;

    // Load the step function from the library
    MTL::Function* stepFunction = ctx.library->newFunction(NS::String::string("rmf_step", NS::UTF8StringEncoding));
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

    // Set the grid buffer
    encoder->setBuffer(ctx.stepGridBuffer, 0, 7);

    MTL::Size threadsPerGrid = MTL::Size(1000, 1, 1); // 40,000 cells, 1,000 threads --> 40 cells per thread
    MTL::Size threadGroupSize = MTL::Size(ctx.pipelineStateStep->maxTotalThreadsPerThreadgroup(), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadGroupSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

int main(int argc, const char* argv[]) {

    // ------------------- Parse Command Line Arguments -------------------
    Params params = parseArgs(argc, const_cast<char**>(argv));

    std::cout << "MCS: " << params.MCS << "\n";
    std::cout << "Lattice Length: " << params.L << "\n";
    std::cout << "Lattice Height: " << params.H << "\n";
    std::cout << "Dimensions: " << params.dimensions << "\n";
    std::cout << "Neighbourhood: " << params.neighbourhood << "\n";
    std::cout << "Print Frequency: " << params.printFrequency << "\n";
    std::cout << "Mobility: " << params.mobility << "\n";

    int MCS = params.MCS;
    int L = params.L;                                      // Length of lattice
    int H = params.H;                                      // Height of lattice
    int N = L * H;                                         // Elementary time steps = total number of cells
    bool moore = params.neighbourhood == 8 ? true : false; // Moore neighbourhood if true, Von Neumann if false

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

    initMetalStep(metalCtx, stepCtx, N); // Initialise Metal step buffers and pipelines

    float M = params.mobility; // Mobility 'since it is proportional to the typical area
                               // explored by one mobile individual per unit time'

    int* grid = new int[N]; // Flattened grid size = L x L

    float mu = 1;    // RPSLS interaction
    float sigma = 1; // Reproduction

    float epsilon = M / (2.0f * N); // Epsilon definition in RMF 2007

    // Normalise the action probabilities
    float sum = mu + sigma + epsilon;
    mu /= sum;
    sigma /= sum;
    epsilon /= sum;

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 3); // Range: 1 to 5 (RPSLS)

    // Randomly initialise the grid with species (stored as int)
    for (int i = 0; i < N; i++) {
        grid[i] = dist(gen); // Randomly assign a species (or EMPTY) as an integer
    }

    // ------------------- Start Simulating -------------------

    for (int mcs = 0; mcs <= MCS; mcs++) {                                 // Monte Carlos
        densities(grid, N, mcs, gridCtx, metalCtx, params.printFrequency); // Every MCS, call densities to add to density vectors for visualisation after simulation
        if (mcs == 0 || mcs == 2000 || mcs == 6000 || mcs % 10000 == 0) {
            plot_snapshot(grid, L, H, moore, mcs, M); // Plot snapshots of the grid at given MCS
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

    show(gridCtx); // Plot density against steps

    std::cout << "Simulation Complete.";

    destroyMetalContext(metalCtx);
    delete[] action_probabilities;
    delete[] cells;
    delete[] neighbours;

    return 0;
}
