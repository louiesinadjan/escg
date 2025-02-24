#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <getopt.h>
#include <iostream>
#include <vector>

//------------------------------------------------------------------------------
// Structure to hold the input parameters of the simulation
//------------------------------------------------------------------------------
struct Params {
    int MCS = 100000;         // Monte Carlo Steps
    int L = 200;              // Length of lattice
    int H = 200;              // Height of lattice
    int dimensions = 2;       // 1D, 2D, 3D
    int neighbourhood = 4;    // Von Neumann (4-way), Moore (8-way)
    int printFrequency = 200; // MCS frequency to print snapshots
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
    uint* cells;
    uint* neighbour_dirs;
    float* action_probabilities;
};

//------------------------------------------------------------------------------
// Parse command line arguments
//------------------------------------------------------------------------------
Params parseArgs(int argc, char* argv[]) {
    Params params; // Uses default values

    // Define long options
    static struct option long_options[] = {
        {"mcs", required_argument, 0, 'm'},
        {"length", required_argument, 0, 'l'},
        {"height", required_argument, 0, 'h'},
        {"dimensions", required_argument, 0, 'd'},
        {"printFrequency", required_argument, 0, 'p'},
        {"neighbourhood", required_argument, 0, 'n'},
        {0, 0, 0, 0} // Terminate options
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "m:l:h:d:p:n:", long_options, &option_index)) != -1) {
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
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [--mcs <MCS>] [--length <Lattice Length>] [--height <Lattice Height>] "
                             "[--boundary <Boundary Type>] [--neighbourhood <Neighbourhood Type>]\n";
                exit(EXIT_FAILURE);
        }
    }

    return params;
}

//------------------------------------------------------------------------------
// CUDA Kernel: Generate random numbers for action_probabilities, cells, and neighbours
//------------------------------------------------------------------------------
__global__ void refreshRandomNumbers(float* action_probabilities, uint32_t* cells, uint32_t* neighbours, int numRandomNumbers, int N, bool moore) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id * 10000 >= numRandomNumbers) {
        return; // Ensure we do not go out of bounds
    }

    // Initialise CURAND random state
    curandState state;
    curand_init(clock64(), id, 0, &state);

    // Each thread generate 10,000 random numbers
    for (int i = 0; i < 10000; i++) {
        action_probabilities[id * 10000 + i] = curand_uniform(&state);                // [0,1]
        cells[id * 10000 + i] = curand(&state) % N;                                   // [0, N-1]
        neighbours[id * 10000 + i] = moore ? curand(&state) % 8 : curand(&state) % 4; // [0, 7] or [0, 3]
    }
}

//------------------------------------------------------------------------------
// Host function to call the CUDA kernel
//------------------------------------------------------------------------------
void generateRandomNumbers(float* d_action_probabilities, uint32_t* d_cells, uint32_t* d_neighbours, int N, int numRandomNumbers, bool moore) {
    int numThreads = numRandomNumbers / 10000;                                // Reduced thread count
    int threadsPerBlock = 256;                                                // Recommended block size
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock; // Compute number of blocks

    std::cout << "Launching CUDA Kernel with " << blocksPerGrid << " blocks and " << threadsPerBlock << " threads per block.\n";

    refreshRandomNumbers<<<blocksPerGrid, threadsPerBlock>>>(d_action_probabilities, d_cells, d_neighbours, numRandomNumbers, N, moore);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed: " << cudaGetErrorString(err) << std::endl;
    }
}

//------------------------------------------------------------------------------
// CUDA Kernel: Populate the grid with random species
//------------------------------------------------------------------------------
__global__ void populateGrid(uint* lattice, int N, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= N) {
        return; // Ensure we do not go out of bounds
    }

    // Initialise CURAND random state
    curandState state;
    curand_init(seed + id, id, 0, &state);

    lattice[id] = (curand(&state) % 5) + 1; // [1, 5]
}

//------------------------------------------------------------------------------
// CUDA Kernel: Compute Densities
//------------------------------------------------------------------------------
__global__ void compute_densities(const uint* __restrict__ grid, // Flattened grid array (40000 elements)
                                  int* __restrict__ result,      // Global output: 6 atomic ints
                                  int gridSize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= gridSize) {
        return; // Ensure we do not go out of bounds
    }

    int species = grid[id];

    if (species >= 0 && species <= 5) {
        atomicAdd(&result[species], 1);
    }
}

//------------------------------------------------------------------------------
// Calculate densities of each species and add to vectors for visualisation
//------------------------------------------------------------------------------
void densities(uint* grid, int N, int mcs, int printFrequency, GridContext& gridCtx) {
    int* d_speciesCounts;
    cudaMalloc(&d_speciesCounts, 6 * sizeof(int));
    cudaMemset(d_speciesCounts, 0, 6 * sizeof(int)); // Initialise to 0

    cudaDeviceSynchronize(); // Ensure all copies are completed

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    compute_densities<<<blocksPerGrid, threadsPerBlock>>>(grid, d_speciesCounts, N);
    cudaDeviceSynchronize(); // Wait for kernel execution to complete

    // Copy results back to host
    int speciesCounts[6];
    cudaMemcpy(speciesCounts, d_speciesCounts, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_speciesCounts);

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

int main(int argc, const char* argv[]) {

    // ------------------- Parse Command Line Arguments -------------------

    Params params = parseArgs(argc, const_cast<char**>(argv));
    std::cout << "MCS: " << params.MCS << "\n";
    std::cout << "Lattice Length: " << params.L << "\n";
    std::cout << "Lattice Height: " << params.H << "\n";
    std::cout << "Dimensions: " << params.dimensions << "\n";
    std::cout << "Neighbourhood: " << params.neighbourhood << "\n";
    std::cout << "Print Frequency: " << params.printFrequency << "\n";

    uint N = params.L * params.H;                          // Size of lattice and number of elementary steps per MCS
    bool moore = params.neighbourhood == 8 ? true : false; // Moore neighbourhood if true, Von Neumann if false

    // ------------------- Random Numbers -------------------

    // Allocate memory on GPU
    uint index = 0;
    const uint numRandomNumbers = 1e7; // 10 million random numbers - lowest value the the GPU does not time out

    // Device pointers
    float* d_action_probabilities;
    uint32_t* d_cells;
    uint32_t* d_neighbours;

    // Host pointers
    float* h_action_probabilities = new float[numRandomNumbers];
    uint32_t* h_cells = new uint32_t[numRandomNumbers];
    uint32_t* h_neighbours = new uint32_t[numRandomNumbers];

    cudaError_t err;

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "GPU Memory Available: " << freeMem / 1024 / 1024 << " MB\n";

    err = cudaMalloc(&d_action_probabilities, numRandomNumbers * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for action_probabilities: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaMalloc(&d_cells, numRandomNumbers * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for cells: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaMalloc(&d_neighbours, numRandomNumbers * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for neighbours: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Generate random numbers on GPU
    generateRandomNumbers(d_action_probabilities, d_cells, d_neighbours, N, numRandomNumbers, moore);

    // Allocate memory on host for verification
    cudaMemcpy(h_action_probabilities, d_action_probabilities, numRandomNumbers * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells, d_cells, numRandomNumbers * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighbours, d_neighbours, numRandomNumbers * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free memory

    // ------------------- Lattice -------------------

    uint* h_grid = new uint[N]; // Allocate memory on host
    uint* d_grid;

    cudaMalloc(&d_grid, N * sizeof(uint)); // Allocate memory on device

    unsigned long long seed = time(NULL); // Generate a unique seed based on current time
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    populateGrid<<<blocksPerGrid, threadsPerBlock>>>(d_grid, N, seed);
    cudaDeviceSynchronize();

    cudaMemcpy(h_grid, d_grid, N * sizeof(uint), cudaMemcpyDeviceToHost);

    // Debug: Print first 20 elements
    for (int i = 0; i < 20; i++) {
        std::cout << "h_grid[" << i << "] = " << h_grid[i] << std::endl;
    }

    // ------------------- Simulation Parameters -------------------

    // The cells to be processed in the current MCS
    // StepContext stepContext;
    // stepContext.cells = new uint[N];
    // stepContext.neighbour_dirs = new uint[N];
    // stepContext.action_probabilities = new float[N];

    GridContext gridContext;

    float M = 1e-6f; // Mobility 'since it is proportional to the typical area
                     // explored by one mobile individual per unit time'

    int* grid = new int[N]; // Flattened grid size = L x L

    float mu = 1;              // RPSLS interaction
    float sigma = 1;           // Reproduction
    float epsilon = 2 * M * N; // Migration

    // Normalise the action probabilities
    float sum = mu + sigma + epsilon;
    mu /= sum;
    sigma /= sum;
    epsilon /= sum;

    // ------------------- Simulation -------------------

    densities(h_grid, N, 0, params.printFrequency, gridContext); // Initial densities

    for (int mcs = 0; mcs <= params.MCS; mcs++) { // Monte Carlos
    }
    // ------------------- Free Memory -------------------

    delete[] h_grid;
    cudaFree(d_grid);

    delete[] h_action_probabilities;
    delete[] h_cells;
    delete[] h_neighbours;
    cudaFree(d_action_probabilities);
    cudaFree(d_cells);
    cudaFree(d_neighbours);

    return 0;
}