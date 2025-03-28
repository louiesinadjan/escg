#include "kernels.cuh"

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

__global__ void compute_densities(const int* __restrict__ grid, int* __restrict__ result, int gridSize, int speciesNum) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= gridSize || id < 0) {
        return;
    }

    // Read species from grid while handling race conditions
    int species = grid[id]; // (Fastest, since no modifications)

    if (species < 0 || species > speciesNum) {
        printf("Error: Invalid species value %d at index %d\n", species, id);
        return; // Skip invalid species
    }

    atomicAdd(&result[species], 1);
}

__device__ bool dominates(int specie, int neighbour, int speciesNum, int* dominance) {
    if (specie == 0 || neighbour == 0) {
        return false; // Empty spaces don't dominate anything
    }
    return dominance[((specie - 1) * speciesNum + (neighbour - 1))] == 1;
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

__global__ void cuda_step(int* grid, int* dominance, int* cells, int* neighbour_dirs, float* action_probs, float mu, float sigma, int L, int H, int speciesNum) {
    int N = L * H;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N || id < 0) {
        return; // Ensure we do not go out of bounds
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

    int cell_index = cells[id];

    if (cell_index < 0 || cell_index >= N) {
        printf("Error: Invalid cell index: %d,  at index: %d\n", cell_index, id);
    }

    int specie = atomicAdd(&grid[cell_index], 0); // Read species safely
    int act = action(action_probs[id], mu, sigma);
    int n_dir = neighbour_dirs[id];

    // Convert 1D index to 2D coordinates
    int row = cell_index / L;
    int col = cell_index % L;

    // Compute neighbour position with wrapping
    int n_row = (row + offsets[n_dir][0] + H) % H;
    int n_col = (col + offsets[n_dir][1] + L) % L;
    int neighbour_index = n_row * L + n_col;

    int neighbour_specie = atomicAdd(&grid[neighbour_index], 0);

    if (act == 1) { // Interaction
        if (dominates(specie, neighbour_specie, speciesNum, dominance)) {
            atomicExch(&grid[neighbour_index], 0); // Remove neighbour
        } else if (dominates(neighbour_specie, specie, speciesNum, dominance)) {
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

__global__ void max_cuda_step(int* grid, int* dominance, int* cells, int* neighbour_dirs, float* action_probs, float mu, float sigma, int L, int H, int speciesNum, int numRandoms) {
    int N = L * H;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int index = global_tid; index < numRandoms; index += totalThreads) {
        int cell_index = cells[index];

        if (cell_index < 0 || cell_index >= N) {
            printf("Error: Invalid cell index: %d, at random index: %d\n", cell_index, index);
            continue;
        }

        int specie = atomicAdd(&grid[cell_index], 0); // Read species safely
        int act = action(action_probs[index], mu, sigma);
        int n_dir = neighbour_dirs[index];

        const int offsets[8][2] = {
            {-1, 0}, {1, 0}, {0, -1}, {0, 1},
            {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
        };

        int row = cell_index / L;
        int col = cell_index % L;

        int n_row = (row + offsets[n_dir][0] + H) % H;
        int n_col = (col + offsets[n_dir][1] + L) % L;
        int neighbour_index = n_row * L + n_col;

        int neighbour_specie = atomicAdd(&grid[neighbour_index], 0);

        if (act == 1) {
            if (dominates(specie, neighbour_specie, speciesNum, dominance)) {
                atomicExch(&grid[neighbour_index], 0);
            } else if (dominates(neighbour_specie, specie, speciesNum, dominance)) {
                atomicExch(&grid[cell_index], 0);
            }
        } else if (act == 2) {
            if (neighbour_specie == 0 && specie != 0) {
                atomicExch(&grid[neighbour_index], specie);
            } else if (specie == 0 && neighbour_specie != 0) {
                atomicExch(&grid[cell_index], neighbour_specie);
            }
        } else if (act == 3) {
            int temp = atomicExch(&grid[cell_index], neighbour_specie);
            atomicExch(&grid[neighbour_index], temp);
        }
    }
}