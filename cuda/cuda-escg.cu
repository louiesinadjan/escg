#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <getopt.h>
#include <iostream>

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
    const uint numRandomNumbers = 1e7;
    float* d_action_probabilities;
    uint32_t* d_cells;
    uint32_t* d_neighbours;

    cudaError_t err;

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
    float* h_action_probabilities = new float[numRandomNumbers];
    uint32_t* h_cells = new uint32_t[numRandomNumbers];
    uint32_t* h_neighbours = new uint32_t[numRandomNumbers];

    // Print some random values to verify

    for (uint i = 0; i < numRandomNumbers; i++) {
        if (h_action_probabilities[i] < 0.0 || h_action_probabilities[i] > 1.0) {
            std::cout << "Error: Action out of bounds " << h_action_probabilities[i] << " | i = " << i << std::endl;
        } else if (h_cells[i] >= N) {
            std::cout << "Error: Cell index out of bounds\n" << h_cells[i] << " | i = " << i << std::endl;
        } else if (moore && (h_neighbours[i] > 7)) {
            std::cout << "Error: Neighbour index out of bounds\n" << h_neighbours[i] << " | i = " << i << std::endl;
        } else if (!moore && (h_neighbours[i] > 3)) {
            std::cout << "Error: Neighbour index out of bounds\n" << h_neighbours[i] << " | i = " << i << std::endl;
        }
    }

    // Free memory
    delete[] h_action_probabilities;
    delete[] h_cells;
    delete[] h_neighbours;

    cudaFree(d_action_probabilities);
    cudaFree(d_cells);
    cudaFree(d_neighbours);

    return 0;
}