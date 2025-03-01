#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
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
// Structure to hold the step data to pass into the CUDA Kernel
//------------------------------------------------------------------------------
struct StepContext {
    int* cells;
    int* neighbour_dirs;
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
// Exports the grid to a CSV file for visualisation in Python
//------------------------------------------------------------------------------
void exportGridToCSV(int* h_grid, Params p, int mcs) {
    std::cout << "(h_grid in export) Grid[0]: " << h_grid[0] << ",\t\tmcs: " << mcs << std::endl;

    std::ostringstream filename;
    std::string nbrhd = p.neighbourhood == 4 ? "nbrhdVN" : "nbrhdM";
    filename << "l" << p.L << "_" << "h" << p.H << "_" << nbrhd << "_mcs" << mcs << ".csv";

    std::ofstream file(filename.str());
    if (!file) {
        std::cerr << "Error: Could not open file " << filename.str() << " for writing." << std::endl;
        return;
    }

    // Write grid data row-wise
    for (int i = 0; i < p.H; i++) {
        for (int j = 0; j < p.L; j++) {
            file << h_grid[i * p.L + j]; // Convert 2D index to 1D
            if (j < p.L - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Grid snapshot saved: " << filename.str() << std::endl;
}

//------------------------------------------------------------------------------
// CUDA Kernel: Generate random numbers for action_probabilities, cells, and neighbours
//------------------------------------------------------------------------------
__global__ void refreshRandomNumbers(float* action_probabilities, int* cells, int* neighbours, int numRandomNumbers, int N, bool moore) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id * 10000 >= numRandomNumbers) {
        return; // Ensure we do not go out of bounds
    }

    // Initialise CURAND random state
    curandState state;
    curand_init(clock64(), id, 0, &state);

    // Each thread generate 10,000 random numbers
    for (int i = 0; i < 10000; i++) {
        int index = id * 10000 + i;

        if (index < numRandomNumbers) {
            action_probabilities[index] = curand_uniform(&state);                // [0,1]
            cells[index] = curand(&state) % N;                                   // [0, N-1]
            neighbours[index] = moore ? curand(&state) % 8 : curand(&state) % 4; // [0, 7] or [0, 3]}
        }
    }
}

//------------------------------------------------------------------------------
// Host function to call the CUDA kernel - Copies random numbers from device to host arrays
//------------------------------------------------------------------------------
void generateRandomNumbers(float* h_action_probabilities, int* h_cells, int* h_neighbours, float* d_action_probabilities, int* d_cells, int* d_neighbours, int N, int numRandomNumbers, bool moore) {
    int numThreads = numRandomNumbers / 10000;                                // Reduced thread count
    int threadsPerBlock = 256;                                                // Recommended block size
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock; // Compute number of blocks

    std::cout << "Launching CUDA Kernel with " << blocksPerGrid << " blocks and " << threadsPerBlock << " threads per block.\n";

    refreshRandomNumbers<<<blocksPerGrid, threadsPerBlock>>>(d_action_probabilities, d_cells, d_neighbours, numRandomNumbers, N, moore);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (refreshRandomNumbers): " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (refreshRandomNumbers): " << cudaGetErrorString(err) << std::endl;
    }

    // Copy memory device -> host
    cudaMemcpy(h_action_probabilities, d_action_probabilities, numRandomNumbers * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells, d_cells, numRandomNumbers * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighbours, d_neighbours, numRandomNumbers * sizeof(int), cudaMemcpyDeviceToHost);
}

//------------------------------------------------------------------------------
// CUDA Kernel: Populate the grid with random species
//------------------------------------------------------------------------------
__global__ void populateGrid(int* grid, int N, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= N) {
        return; // Ensure we do not go out of bounds
    }

    // Initialise CURAND random state
    curandState state;
    curand_init(seed + id, id, 0, &state);

    grid[id] = (curand(&state) % 5) + 1; // [1, 5]
}

//------------------------------------------------------------------------------
// CUDA Kernel: Compute Densities
//------------------------------------------------------------------------------
__global__ void compute_densities(const int* __restrict__ grid, int* __restrict__ result, int gridSize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= gridSize)
        return;

    int species = grid[id];

    if (species < 0 || species > 5) {
        printf("Error: Invalid species value %d at index %d\n", species, id);
        return; // Skip invalid species
    }

    atomicAdd(&result[species], 1);
}

//------------------------------------------------------------------------------
// Calculate densities of each species and add to vectors for visualisation
//------------------------------------------------------------------------------
void densities(int* grid, int N, int mcs, int printFrequency, GridContext& gridCtx) {
    int* d_speciesCounts;
    cudaMalloc(&d_speciesCounts, 6 * sizeof(int));
    cudaMemset(d_speciesCounts, 0, 6 * sizeof(int)); // Initialise to 0

    // Copy grid before launching to the kernel
    int* d_grid;
    cudaMalloc(&d_grid, N * sizeof(int));
    cudaMemcpy(d_grid, grid, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    compute_densities<<<blocksPerGrid, threadsPerBlock>>>(d_grid, d_speciesCounts, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (compute_densities): " << cudaGetErrorString(err) << std ::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (compute_densities): " << cudaGetErrorString(err) << std ::endl;
    }

    // Copy results back to host
    int speciesCounts[6];
    cudaMemcpy(speciesCounts, d_speciesCounts, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_speciesCounts);
    cudaFree(d_grid);

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
        // Raw counts
        // std::cout << "Population Densities at: " << mcs << std::endl;
        // std::cout << "EMPTY: " << speciesCounts[0] << std::endl;
        // std::cout << "ROCK: " << speciesCounts[1] << std::endl;
        // std::cout << "PAPER: " << speciesCounts[2] << std::endl;
        // std::cout << "SCISSORS: " << speciesCounts[3] << std::endl;
        // std::cout << "LIZARD: " << speciesCounts[4] << std::endl;
        // std::cout << "SPOCK: " << speciesCounts[5] << std::endl;

        std::cout << std::endl;
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
// Initialise the grid with random species
//------------------------------------------------------------------------------
void initialiseGrid(int*& d_grid, int*& h_grid, int N) {
    // Allocate memory on host before copying from GPU
    h_grid = new int[N];

    // Allocate memory on device
    cudaMalloc(&d_grid, N * sizeof(int));

    unsigned long long seed = time(NULL); // Generate a unique seed based on current time
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    populateGrid<<<blocksPerGrid, threadsPerBlock>>>(d_grid, N, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (populateGrid): " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (populateGrid): " << cudaGetErrorString(err) << std ::endl;
    }

    // Copy from device to host
    cudaMemcpy(h_grid, d_grid, N * sizeof(int), cudaMemcpyDeviceToHost);
}

__device__ bool dominates(int specie, int neighbour) {
    switch (specie) {
        case 1:                    // ROCK (crushes 3: SCISSORS, crushes 4: LIZARD)
            return neighbour == 4; // Absence of Rock - Scissors interaction
        case 2:                    // PAPER (covers 1: ROCK, disproves 5: SPOCK)
            return (neighbour == 1 || neighbour == 5);
        case 3: // SCISSORS (cuts 2: PAPER, decapitates 4: LIZARD)
            return (neighbour == 2 || neighbour == 4);
        case 4: // LIZARD (poisons 5: SPOCK, eats 2: PAPER)
            return (neighbour == 5 || neighbour == 2);
        case 5: // SPOCK (smashes 3: SCISSORS, vaporises 1: ROCK)
            return (neighbour == 3 || neighbour == 1);
        default: // 0 (EMPTY) or any invalid integer
            return false;
    }
}

__device__ int action(float action_prob, float mu, float sigma) {
    if (action_prob < mu) {
        return 1; // Interaction
    } else if (action_prob < mu + sigma) {
        return 2; // Reproduction
    } else {
        return 3; // Migration
    }
}

__global__ void cuda_step(int* grid, int* cells, int* neighbour_dirs, float* action_probs, float mu, float sigma, int L, int H) {
    int N = L * H;
    int totalThreads = gridDim.x * blockDim.x;
    int cellsPerThread = (N + totalThreads - 1) / totalThreads;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // if (id == 0) {
    //     printf("Kernel has started execution!\n");
    //     printf("cellsperthread = %d", cellsPerThread);
    // }

    if (id == 0) {
        atomicExch(&grid[0], 99);
    }

    // Precompute offsets for neighbour directions
    const int offsets[8][2] = {
        {-1, 0},  // Up
        {1, 0},   // Down
        {0, -1},  // Left
        {0, 1},   // Right
        {-1, -1}, // Up-Left
        {-1, 1},  // Up-Right
        {1, -1},  // Down-Left
        {1, 1}    // Down-Right
    };

    for (int i = 0; i < cellsPerThread; i++) {
        int index = id * cellsPerThread + i;

        if (index >= N) {
            printf("Error: Invalid index: %d\n", index);
            return; // Ensure we don't go out of bounds
        }

        int cell_index = cells[index];
        if (cell_index == -1 || cell_index >= N) {
            printf("Error: Invalid cell index: %d,  at index: %d\n", cell_index, index);
            continue; // Skip invalid cells
        }

        int specie = atomicAdd(&grid[cell_index], 0); // Read species safely
        int act = action(action_probs[index], mu, sigma);
        int n_dir = neighbour_dirs[index];

        // Convert 1D index to 2D coordinates
        int row = cell_index / L;
        int col = cell_index % L;

        // Compute neighbour position with wrapping
        int n_row = (row + offsets[n_dir][0] + H) % H;
        int n_col = (col + offsets[n_dir][1] + L) % L;
        int neighbour_index = n_row * L + n_col;

        int neighbour_specie = atomicAdd(&grid[neighbour_index], 0);

        if (act == 1) { // Interaction
            if (dominates(specie, neighbour_specie)) {
                atomicExch(&grid[neighbour_index], 0); // Remove neighbour
            } else if (dominates(neighbour_specie, specie)) {
                atomicExch(&grid[cell_index], 0); // Remove self
            }
        } else if (act == 2) { // Reproduction
            if (neighbour_specie == 0 && specie != 0) {
                atomicExch(&grid[neighbour_index], specie); // Reproduce into empty neighbour
            } else if (specie == 0 && neighbour_specie != 0) {
                atomicExch(&grid[cell_index], neighbour_specie);
            }
        } else if (act == 3) { // Migration (swap species)
            int temp = atomicExch(&grid[cell_index], neighbour_specie);
            atomicExch(&grid[neighbour_index], temp);
        }
    }
}

void step(int* h_grid, int* d_grid, int* step_cells, int* step_neighbour_dirs, float* step_action_probabilities, float mu, float sigma, int L, int H) {
    int N = L * H;

    cudaMemcpy(d_grid, h_grid, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (L * H + threadsPerBlock - 1) / threadsPerBlock;
    cuda_step<<<blocksPerGrid, threadsPerBlock>>>(d_grid, step_cells, step_neighbour_dirs, step_action_probabilities, mu, sigma, L, H);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (cuda_step): " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (cuda_step): " << cudaGetErrorString(err) << std ::endl;
    }

    // Copy from device to host
    cudaMemcpy(h_grid, d_grid, N * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "(h_grid in step) Grid[0]: " << h_grid[0] << std::endl;
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

    int N = params.L * params.H;                           // Size of lattice and number of elementary steps per MCS
    bool moore = params.neighbourhood == 8 ? true : false; // Moore neighbourhood if true, Von Neumann if false

    // ------------------- Random Numbers -------------------

    // Allocate memory on GPU
    int index = 0;
    const int numRandomNumbers = 5e7; // 50 million random numbers - lowest value the the GPU does not time out
    // const int numRandomNumbers = 1e6; // 50 million random numbers - lowest value the the GPU does not time out

    // Device pointers
    float* d_action_probabilities;
    int* d_cells;
    int* d_neighbours;

    // Host pointers
    float* h_action_probabilities = new float[numRandomNumbers];
    int* h_cells = new int[numRandomNumbers];
    int* h_neighbours = new int[numRandomNumbers];

    cudaError_t err;

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "GPU Memory Available: " << freeMem / 1024 / 1024 << " MB\n";

    err = cudaMalloc(&d_action_probabilities, numRandomNumbers * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for action_probabilities: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMalloc(&d_cells, numRandomNumbers * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for cells: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMalloc(&d_neighbours, numRandomNumbers * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for neighbours: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Generate random numbers on GPU
    generateRandomNumbers(h_action_probabilities, h_cells, h_neighbours, d_action_probabilities, d_cells, d_neighbours, N, numRandomNumbers, moore);

    // ------------------- Lattice -------------------

    int* h_grid; // Allocate memory on host
    int* d_grid;

    initialiseGrid(d_grid, h_grid, N);

    // ------------------- Simulation Parameters -------------------

    // The cells to be processed in the current MCS
    StepContext stepContext;
    stepContext.cells = new int[N];
    stepContext.neighbour_dirs = new int[N];
    stepContext.action_probabilities = new float[N];

    GridContext gridContext;

    float M = 1e-6f; // Mobility 'since it is proportional to the typical area
                     // explored by one mobile individual per unit time'

    // grid is stored on h_grid

    float mu = 1;              // RPSLS interaction
    float sigma = 1;           // Reproduction
    float epsilon = 2 * M * N; // Migration

    // Normalise the action probabilities
    float sum = mu + sigma + epsilon;
    mu /= sum;
    sigma /= sum;
    epsilon /= sum;

    // ------------------- Simulation -------------------

    exportGridToCSV(h_grid, params, 0);
    for (int mcs = 0; mcs <= params.MCS; mcs++) { // Monte Carlos
        densities(h_grid, N, mcs, params.printFrequency, gridContext);

        exportGridToCSV(h_grid, params, mcs);

        // Export grid to csv for visualisation
        if (mcs == 2000 || mcs == 6000 || mcs == 20000 || mcs == 100000) {
            exportGridToCSV(h_grid, params, mcs);
        }

        for (int i = 0; i < N; i++) {
            if (index >= numRandomNumbers) {
                generateRandomNumbers(h_action_probabilities, h_cells, h_neighbours, d_action_probabilities, d_cells, d_neighbours, N, numRandomNumbers, moore);
                index = 0;
            }

            stepContext.cells[i] = h_cells[index];
            stepContext.neighbour_dirs[i] = h_neighbours[index];
            stepContext.action_probabilities[i] = h_action_probabilities[index];
            index++;
        }

        step(h_grid, d_grid, stepContext.cells, stepContext.neighbour_dirs, stepContext.action_probabilities, mu, sigma, params.L, params.H);
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