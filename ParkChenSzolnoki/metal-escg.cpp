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
#include <set>
#include <sstream>
#include <vector>

Params parseArgs(int argc, char* argv[]) {
    Params params; // Uses default values

    static struct option long_options[] = {
        {"mcs", required_argument, 0, 'm'},
        {"length", required_argument, 0, 'l'},
        {"height", required_argument, 0, 'h'},
        {"printFrequency", required_argument, 0, 'p'},
        {"neighbourhood", required_argument, 0, 'n'},
        {"species", required_argument, 0, 's'},
        {"mobility", required_argument, 0, 'M'},
        {"flux", required_argument, 0, 'f'},
        {"empty", required_argument, 0, 'w'},
        {"dominance", required_argument, 0, 'd'},
        {"save", required_argument, 0, 's'},
        {"resume", required_argument, 0, 'r'},
        {"numRandoms", required_argument, 0, 'R'},
        {"maxStep", required_argument, 0, 'x'},

        {"alpha", required_argument, 0, 'a'},
        {"beta", required_argument, 0, 'b'},
        {"gamma", required_argument, 0, 'g'},

        {0, 0, 0, 0} // End of options
    };

    int opt;
    int option_index = 0;

    // Dominance refers to importing a dominance.csv as the dominance adjacency matrix
    std::string usage = "[--mcs <MCS>] [--length <Lattice Length>] [--height <Lattice Height>] "
                        "[--printFrequency <Print Frequency>] [--empty <Initial Empty Cell Probability >] "
                        "[--neighbourhood <Neighbourhood 4/8>] [--mobility <float>] "
                        "[--species <int>] [--flux <true|false>] [--dominance <true|false]"
                        "[--numRandoms <int>][--maxStep <true|false]"
                        "[--save <true|false>] [--resume <true|false>]"
                        "[--alpha <float>] [--beta <float>] [--gamma <float>]";

    // Parse the command line arguments
    while ((opt = getopt_long(argc, argv, "m:l:h:p:n:M:s:f:e:r:d:S:x:R:a:b:g:", long_options, &option_index)) != -1) {
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
                if (params.species < 0) {
                    std::cerr << "Error: Number of species must be >0." << std::endl;
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
            case 'd': {
                std::string domStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : domStr) {
                    c = tolower(c);
                }
                if (domStr == "true" || domStr == "1" || domStr == "yes") {
                    params.dominance = true;
                } else {
                    params.dominance = false;
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
            case 'S': {
                std::string saveStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : saveStr) {
                    c = tolower(c);
                }
                if (saveStr == "true" || saveStr == "1" || saveStr == "yes") {
                    params.save = true;
                } else {
                    params.save = false;
                }
                break;
            }
            case 'x': {
                std::string maxStepStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : maxStepStr) {
                    c = tolower(c);
                }
                if (maxStepStr == "false" || maxStepStr == "0" || maxStepStr == "no") {
                    params.maxStep = false;
                } else {
                    params.maxStep = true;
                }
                break;
            }
            case 'R':
                params.numRandoms = std::stof(optarg);
                if (params.numRandoms < 1000000) {
                    std::cout << "Entered: " << params.numRandoms << std::endl;
                    std::cerr << "Error: Random numbers must be at least 1,000,000." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'a':
                params.alpha = std::stof(optarg);
                break;
            case 'b':
                params.beta = std::stof(optarg);
                break;
            case 'g':
                params.gamma = std::stof(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << usage << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    return params;
}

bool initMetalContext(MetalContext& ctx, int N, int randomNums, int speciesNum) {
    // 100,000,000  random numbers per shader
    ctx.threads = randomNums / 10000; // 10,000 numbers per thread

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
    NS::String* libraryPath = NS::String::string("/Users/louiesinadjan/Documents/dissertation/escg/ParkChenSzolnoki/build/escg.metallib", NS::UTF8StringEncoding);

    ctx.library = ctx.device->newLibrary(libraryPath, &error);
    if (!ctx.library) {
        std::cerr << "Error: Failed to load escg.metallib - " << error->localizedDescription()->utf8String() << std::endl;
        return false;
    }

    // Load the shader functions from the library
    // Randoms
    MTL::Function* random_invasions = ctx.library->newFunction(NS::String::string("mt_random_invasions", NS::UTF8StringEncoding));
    MTL::Function* random_cells = ctx.library->newFunction(NS::String::string("mt_random_cells", NS::UTF8StringEncoding));
    MTL::Function* random_neighbours = ctx.library->newFunction(NS::String::string("mt_random_neighbours", NS::UTF8StringEncoding));

    // Densities
    MTL::Function* computeDensities = ctx.library->newFunction(NS::String::string("compute_densities", NS::UTF8StringEncoding));

    if (!random_invasions || !random_cells || !random_neighbours || !computeDensities) {
        std::cerr << "Error: Failed to find one or more functions in Metal library." << std::endl;
        return false;
    }

    // Create compute pipeline states
    // Randoms
    ctx.pipelineStateInvasions = ctx.device->newComputePipelineState(random_invasions, &error);
    ctx.pipelineStateCells = ctx.device->newComputePipelineState(random_cells, &error);
    ctx.pipelineStateNeighbours = ctx.device->newComputePipelineState(random_neighbours, &error);

    // Densities
    ctx.pipelineStateDensities = ctx.device->newComputePipelineState(computeDensities, &error);

    if (!ctx.pipelineStateInvasions || !ctx.pipelineStateCells || !ctx.pipelineStateNeighbours || !ctx.pipelineStateDensities) {
        std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        return false;
    }

    // Prepare the seed buffer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000);
    uint32_t* seeds = new uint32_t[ctx.threads];
    for (int i = 0; i < ctx.threads; i++) {
        dist(gen);
    }
    ctx.seedBuffer = ctx.device->newBuffer(seeds, sizeof(uint32_t) * ctx.threads, MTL::ResourceStorageModeShared);
    delete[] seeds;

    // Create result buffers
    // Randoms
    ctx.resultBufferInvasions = ctx.device->newBuffer(sizeof(float) * randomNums, MTL::ResourceStorageModeShared);
    ctx.resultBufferCells = ctx.device->newBuffer(sizeof(uint32_t) * randomNums, MTL::ResourceStorageModeShared);
    ctx.resultBufferNeighbours = ctx.device->newBuffer(sizeof(uint32_t) * randomNums, MTL::ResourceStorageModeShared);

    // Densities
    ctx.gridBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);                          // Buffer size of L * H
    ctx.densityResultsBuffer = ctx.device->newBuffer(sizeof(int) * (speciesNum + 1), MTL::ResourceStorageModeShared); // speciesNum + 1 --> e.g., E + RPSLS

    return true;
}

//------------------------------------------------------------------------------
// Refresh: Use existing pipeline and buffer objects to generate new random numbers
//------------------------------------------------------------------------------
RandomCommandBuffers refreshRandomNumbers(MetalContext& ctx, int N, bool moore) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000);
    uint32_t* seeds = new uint32_t[ctx.threads];
    for (int i = 0; i < ctx.threads; i++) {
        dist(gen);
    }
    std::memcpy(ctx.seedBuffer->contents(), seeds, sizeof(uint32_t) * ctx.threads);
    delete[] seeds;

    // Setup common dispatch parameters
    MTL::Size gridSize = MTL::Size(ctx.threads, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(ctx.pipelineStateInvasions->maxTotalThreadsPerThreadgroup(), 1, 1);

    // Refresh random actions buffers
    MTL::CommandBuffer* invasionCommandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder1 = invasionCommandBuffer->computeCommandEncoder();
    encoder1->setComputePipelineState(ctx.pipelineStateInvasions);
    encoder1->setBuffer(ctx.seedBuffer, 0, 0);
    encoder1->setBuffer(ctx.resultBufferInvasions, 0, 1);
    encoder1->dispatchThreads(gridSize, threadGroupSize);
    encoder1->endEncoding();
    invasionCommandBuffer->commit(); // Commit without waiting

    // Refresh random cells buffers
    MTL::CommandBuffer* cellCommandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder2 = cellCommandBuffer->computeCommandEncoder();
    encoder2->setComputePipelineState(ctx.pipelineStateCells);
    encoder2->setBuffer(ctx.seedBuffer, 0, 0);
    encoder2->setBuffer(ctx.resultBufferCells, 0, 1);
    encoder2->setBytes(&N, sizeof(int), 2);
    encoder2->dispatchThreads(gridSize, threadGroupSize);
    encoder2->endEncoding();
    cellCommandBuffer->commit(); // Commit without waiting

    // Refresh random neighbours buffers
    MTL::CommandBuffer* neighbourCommandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder3 = neighbourCommandBuffer->computeCommandEncoder();
    encoder3->setComputePipelineState(ctx.pipelineStateNeighbours);
    encoder3->setBuffer(ctx.seedBuffer, 0, 0);
    encoder3->setBuffer(ctx.resultBufferNeighbours, 0, 1);
    encoder3->setBytes(&moore, sizeof(bool), 2);
    encoder3->dispatchThreads(gridSize, threadGroupSize);
    encoder3->endEncoding();
    neighbourCommandBuffer->commit(); // Commit without waiting

    return {cellCommandBuffer, neighbourCommandBuffer, invasionCommandBuffer};
}

void destroyMetalContext(MetalContext& ctx) {
    if (ctx.seedBuffer)
        ctx.seedBuffer->release();
    if (ctx.resultBufferInvasions)
        ctx.resultBufferInvasions->release();
    if (ctx.resultBufferCells)
        ctx.resultBufferCells->release();
    if (ctx.resultBufferNeighbours)
        ctx.resultBufferNeighbours->release();
    if (ctx.pipelineStateInvasions)
        ctx.pipelineStateInvasions->release();
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
    if (ctx.invasionProbabilitiesBuffer)
        ctx.invasionProbabilitiesBuffer->release();
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
void computeDensitiesGPU(MetalContext& ctx, int* grid, int* densities, int N, int speciesNum) {
    // Copy the flattened grid to the GPU buffer
    std::memcpy(ctx.gridBuffer->contents(), grid, sizeof(int) * N);
    // Reset the density buffer
    std::memset(ctx.densityResultsBuffer->contents(), 0, sizeof(int) * (speciesNum));

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
    std::memcpy(densities, ctx.densityResultsBuffer->contents(), sizeof(int) * (speciesNum));
}

void densities(int* grid, int N, int mcs, GridContext& gridCtx, MetalContext& metalCtx, int printFrequency, int speciesNum, std::set<int>& speciesSet) {
    int* speciesCounts = new int[8]; // [species...]
    std::fill_n(speciesCounts, 8, 0);

    for (int i = 0; i < N; i++) {
        speciesCounts[grid[i]]++;
    }

    for (int i = 0; i < speciesNum; i++) {
        if (speciesCounts[i] == 0) {
            speciesSet.erase(i);
        }
    }

    if (mcs % printFrequency == 0) {
        std::cout << "Population Densities at: " << mcs << std::endl;
        for (int i = 0; i < speciesNum; i++) {
            std::cout << "Species " << i << ": " << (static_cast<double>(speciesCounts[i]) / N) * 100 << std::endl;
        }
        std::cout << std::endl;
    }

    delete[] speciesCounts;
}

void show(GridContext& gridCtx, Params params) { plot_densities(gridCtx, params); }

void initMetalStep(MetalContext& ctx, int N, int species, bool maxStep, int numRandoms) {
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

    if (maxStep) { // Process (numRandoms / N) MCS per Metal call
        ctx.cellsBuffer = ctx.device->newBuffer(sizeof(int) * numRandoms, MTL::ResourceStorageModeShared);
        ctx.neighboursDirsBuffer = ctx.device->newBuffer(sizeof(int) * numRandoms, MTL::ResourceStorageModeShared);
        ctx.invasionProbabilitiesBuffer = ctx.device->newBuffer(sizeof(float) * numRandoms, MTL::ResourceStorageModeShared);
    } else { // Process 1 MCS per Metal call
        ctx.cellsBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);
        ctx.neighboursDirsBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);
        ctx.invasionProbabilitiesBuffer = ctx.device->newBuffer(sizeof(float) * N, MTL::ResourceStorageModeShared);
    }

    ctx.stepGridBuffer = ctx.device->newBuffer(sizeof(int) * N, MTL::ResourceStorageModeShared);
    ctx.dominanceBuffer = ctx.device->newBuffer(sizeof(int) * species * species, MTL::ResourceStorageModeShared);
}

// ------------------------------------------------------------------------------
// Metal step shader - (numRandoms / N) MCS per call
// ------------------------------------------------------------------------------
void metalStep(MetalContext& ctx, StepContext stepCtx, int N, Params& p, int* grid, float* dominance) {
    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    // Assign to the buffers
    std::memcpy(ctx.cellsBuffer->contents(), stepCtx.cells, sizeof(int) * N);
    std::memcpy(ctx.neighboursDirsBuffer->contents(), stepCtx.neighbour_dirs, sizeof(int) * N);
    std::memcpy(ctx.invasionProbabilitiesBuffer->contents(), stepCtx.invasion_probabilities, sizeof(float) * N);
    std::memcpy(ctx.stepGridBuffer->contents(), grid, sizeof(int) * N);
    std::memcpy(ctx.dominanceBuffer->contents(), dominance, sizeof(float) * p.species * p.species);

    // Create buffers for cells, neighbour directions, action probabilities
    encoder->setComputePipelineState(ctx.pipelineStateStep);
    encoder->setBuffer(ctx.cellsBuffer, 0, 0);
    encoder->setBuffer(ctx.neighboursDirsBuffer, 0, 1);
    encoder->setBuffer(ctx.invasionProbabilitiesBuffer, 0, 2);
    encoder->setBuffer(ctx.dominanceBuffer, 0, 3);

    // Using `setBytes()` for scalar values
    // encoder->setBytes(&alpha, sizeof(float), 3);
    encoder->setBytes(&p.beta, sizeof(float), 4);
    encoder->setBytes(&p.gamma, sizeof(float), 5);
    encoder->setBytes(&p.L, sizeof(int), 6);
    encoder->setBytes(&p.H, sizeof(int), 7);
    encoder->setBytes(&p.flux, sizeof(bool), 8);
    encoder->setBytes(&p.species, sizeof(bool), 9);
    encoder->setBytes(&p.numRandoms, sizeof(int), 10);
    encoder->setBytes(&p.maxStep, sizeof(bool), 11);

    // Set the grid buffer
    encoder->setBuffer(ctx.stepGridBuffer, 0, 12);

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
    std::uniform_int_distribution<int> dist(0, 7); // Range: 1 to 5 (RPSLS) or p.species

    // Randomly initialise the grid with species (stored as int)
    for (int i = 0; i < p.L * p.H; i++) {
        grid[i] = dist(gen); // Randomly assign a species (or EMPTY) as an integer
    }
}

bool stasis(std::set<int> speciesSet) { return speciesSet.size() <= 1; }

bool compareGrid(int* grid, int* prevGrid, int L, int mcs, Params p) {
    bool same = std::memcmp(grid, prevGrid, sizeof(int) * L * L) == 0;

    if (same) {
        std::vector<int> thresholds = {10, 50, 100, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};
        for (int t : thresholds) {
            if (mcs < t) {
                writeResults(grid, L, t, p.alpha, p.beta, true);
            }
        }
    }

    return same;
}

int main(int argc, const char* argv[]) {
    // ------------------- Parse Command Line Arguments -------------------
    Params params = parseArgs(argc, const_cast<char**>(argv));
    params.species = 8; // Park Chen Szolnoki
    params.H = params.L;
    params.maxStep = false;

    int currentMCS = 0;
    int MCS = params.MCS;
    int L;      // Length of lattice
    int H;      // Height of lattice
    int N;      // Elementary time steps = total number of cells
    bool moore; // Moore neighbourhood if true, Von Neumann if false
    int* grid;  // Flattened grid size = L x H
    int* prevGrid;

    params.numRandoms = (params.numRandoms / N) * N;

    float* dominance;

    // std::cout << "Starting new simulation.\n" << std::endl;

    L = params.L;
    N = L * L;

    MCS = params.MCS;
    moore = params.neighbourhood == 8;
    grid = new int[N];
    prevGrid = new int[N];
    std::memset(prevGrid, -1, sizeof(int) * N);

    // Creating output directory
    std::string length = "l" + std::to_string(L);
    std::string neighbourhood = moore ? "Moore" : "VN";
    std::ostringstream oss;
    oss.precision(2);
    oss << std::scientific << params.mobility;
    std::string mobility_str = "M" + oss.str();
    std::string flux = params.flux ? "flux" : "noflux";
    std::string species = std::to_string(params.species) + "species";
    params.outputDir = length + "_" + neighbourhood + "_" + mobility_str + "_" + flux + "_" + species;

    dominance = new float[64]; // 8x8 matrix
    generateDominance(dominance, params.alpha, params.beta, params.gamma);
    exportDominanceToCSV(dominance, params.species, params);

    if (params.save) {
        // Create the directory
        if (std::filesystem::create_directory(params.outputDir)) {
            std::cout << "Directory created: " << params.outputDir << std::endl;
        } else {
            std::cout << "Failed to create or already exists: " << params.outputDir << std::endl;
        }

        // Export simulation metadata to directory
        exportParamsToCSV(params);                 // Export the parameters to a csv file
        exportGridToCSV(grid, params, currentMCS); // Export the grid to a csv file
        exportDominanceToCSV(dominance, params.species, params);
    }

    // exportDominanceToCSV(dominance, 8, params);

    // ------------------- Metal Parameters -------------------

    float* invasion_probabilities = new float[params.numRandoms];
    uint32_t* cells = new uint32_t[params.numRandoms];
    uint32_t* neighbours = new uint32_t[params.numRandoms];
    int index = 0;

    MetalContext metalCtx;
    if (!initMetalContext(metalCtx, N, params.numRandoms, params.species)) {
        std::cerr << "Failed to initialise Metal context." << std::endl;
        return -1;
    }

    RandomCommandBuffers cmdBuffers = refreshRandomNumbers(metalCtx, N, moore);

    cmdBuffers.cellsCommandBuffer->waitUntilCompleted();
    cmdBuffers.neighboursCommandBuffer->waitUntilCompleted();
    cmdBuffers.invasionsCommandBuffer->waitUntilCompleted();

    std::memcpy(cells, metalCtx.resultBufferCells->contents(), sizeof(uint32_t) * params.numRandoms);
    std::memcpy(neighbours, metalCtx.resultBufferNeighbours->contents(), sizeof(uint32_t) * params.numRandoms);
    std::memcpy(invasion_probabilities, metalCtx.resultBufferInvasions->contents(), sizeof(float) * params.numRandoms);

    // ------------------- Simulation Parameters -------------------

    GridContext gridCtx;

    initMetalStep(metalCtx, N, params.species, params.maxStep, params.numRandoms); // Initialise Metal step buffers and pipelines

    if (params.resume) {
        std::cout << "Resuming simulation from MCS: " << currentMCS << std::endl;
    } else {
        initialiseGrid(grid, params); // Initialise the grid
    }

    std::set<int> speciesSet;
    for (int i = 0; i < params.species; i++) {
        speciesSet.insert(i);
    }

    // ------------------- Start Simulating -------------------

    if (params.save) {
        plot_snapshot(grid, currentMCS, params);   // Plot snapshots at specific MCS
        exportGridToCSV(grid, params, currentMCS); // Export the grid to a csv file
    }

    StepContext stepCtx;
    stepCtx.invasion_probabilities = new float[N];
    stepCtx.cells = new int[N];
    stepCtx.neighbour_dirs = new int[N];

    if (params.resume) {
        importCSVToGrid(grid, N);
    }

    for (int mcs = 0; mcs <= MCS; mcs++) {
        if (compareGrid(grid, prevGrid, L, mcs, params)) {
            std::cout << "Stasis reached at MCS: " << mcs << std::endl;
            break;
        }

        if (mcs <= 10) {
            writeResults(grid, L, mcs, params.alpha, params.beta, false);
            if (params.save) {
                plot_snapshot(grid, mcs, params);
            }
        } else if (mcs == 50 || mcs == 100 || mcs == 1000 || mcs == 5000) {
            writeResults(grid, L, mcs, params.alpha, params.beta, false);
            if (params.save) {
                plot_snapshot(grid, mcs, params);
            }
        } else if (mcs % 10000 == 0) {
            writeResults(grid, L, mcs, params.alpha, params.beta, false);
            if (params.save) {
                plot_snapshot(grid, mcs, params);
            }
        }

        // densities(grid, N, mcs, gridCtx, metalCtx, params.printFrequency, params.species, speciesSet);

        if (mcs == MCS) {
            break;
        }

        std::memcpy(metalCtx.stepGridBuffer->contents(), grid, sizeof(int) * N);

        for (int i = 0; i < N; i++) {
            if (index == 0) {
                cmdBuffers = refreshRandomNumbers(metalCtx, N, moore);
            } else if (index >= params.numRandoms) {
                cmdBuffers.cellsCommandBuffer->waitUntilCompleted();
                cmdBuffers.neighboursCommandBuffer->waitUntilCompleted();
                cmdBuffers.invasionsCommandBuffer->waitUntilCompleted();

                std::memcpy(cells, metalCtx.resultBufferCells->contents(), sizeof(uint32_t) * params.numRandoms);
                std::memcpy(neighbours, metalCtx.resultBufferNeighbours->contents(), sizeof(uint32_t) * params.numRandoms);
                std::memcpy(invasion_probabilities, metalCtx.resultBufferInvasions->contents(), sizeof(float) * params.numRandoms);
                index = 0;
            }

            stepCtx.cells[i] = cells[index];
            stepCtx.neighbour_dirs[i] = neighbours[index];
            stepCtx.invasion_probabilities[i] = invasion_probabilities[index];
            index++;
        }

        metalStep(metalCtx, stepCtx, N, params, grid, dominance);

        std::memcpy(prevGrid, grid, sizeof(int) * N);
        std::memcpy(grid, metalCtx.stepGridBuffer->contents(), sizeof(int) * N);
    }

    if (params.save) {
        show(gridCtx, params); // Plot density against steps
    }

    destroyMetalContext(metalCtx);
    delete[] invasion_probabilities;
    delete[] cells;
    delete[] neighbours;

    delete[] grid;
    delete[] prevGrid;
    delete[] dominance;

    return 0;
}
