#include "kernels.cuh"

__global__ void refreshRandomNumbers(float* invasion_probabilities, int* cells, int* neighbours, int numRandomNumbers, int N, bool moore) {
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
            invasion_probabilities[index] = curand_uniform(&state);              // [0,1]
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

__device__ float dominates(int specie, int neighbour, int speciesNum, float* dominance) { return dominance[(specie * speciesNum + neighbour)]; }

__global__ void cuda_step(int* grid, float* dominance, int* cells, int* neighbour_dirs, float* invasion_probs, int L) {
    int N = L * L;
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

    int i = cells[id];

    if (i < 0 || i >= N) {
        printf("Error: Invalid cell index: %d,  at index: %d\n", i, id);
    }

    int specie = atomicAdd(&grid[i], 0); // Read species safely
    int n_dir = neighbour_dirs[id];
    float invasion = invasion_probs[id];

    // Convert 1D index to 2D coordinates
    int row = i / L;
    int col = i % L;

    // Compute neighbour position with wrapping
    int n_row = (row + offsets[n_dir][0] + L) % L;
    int n_col = (col + offsets[n_dir][1] + L) % L;
    int n_i = n_row * L + n_col;

    int neighbour = atomicAdd(&grid[n_i], 0);

    if (specie == neighbour) {
        return;
    }

    // Let specie be the lower indexed specie
    if (specie > neighbour) {
        int temp = specie; 
        specie = neighbour;
        neighbour = temp;

        int t_i = i;
        i = n_i;
        n_i = t_i;
    }

    float specieDom = dominates(specie, neighbour, 8, dominance);
    float neighbourDom = dominates(neighbour, specie, 8, dominance);

    if (specieDom == neighbourDom) {
        return;
    } else if (invasion < specieDom) {
        atomicExch(&grid[n_i], specie); // attacker wins
    } else if (invasion < neighbourDom) {
        atomicExch(&grid[i], neighbour); // defender wins
    }
}

__global__ void max_cuda_step(int* grid, float* dominance, int* cells, int* neighbour_dirs, float* invasion_probs, int L, int numRandoms) {
    int N = L * L;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

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


    for (int id = global_tid; id < numRandoms; id += totalThreads) {
        //  //  //  //
        int i = cells[id];

        if (i < 0 || i >= N) {
            printf("Error: Invalid cell index: %d,  at index: %d\n", i, id);
        }

        int specie = atomicAdd(&grid[i], 0); // Read species safely
        int n_dir = neighbour_dirs[id];
        float invasion = invasion_probs[id];

        // Convert 1D index to 2D coordinates
        int row = i / L;
        int col = i % L;

        // Compute neighbour position with wrapping
        int n_row = (row + offsets[n_dir][0] + L) % L;
        int n_col = (col + offsets[n_dir][1] + L) % L;
        int n_i = n_row * L + n_col;

        int neighbour = atomicAdd(&grid[n_i], 0);

        if (specie == neighbour) {
            return;
        }

        // Let specie be the lower indexed specie
        if (specie > neighbour) {
            int temp = specie; 
            specie = neighbour;
            neighbour = temp;

            int t_i = i;
            i = n_i;
            n_i = t_i;
        }

        float specieDom = dominates(specie, neighbour, 8, dominance);
        float neighbourDom = dominates(neighbour, specie, 8, dominance);

        if (specieDom == neighbourDom) {
            return;
        } else if (invasion < specieDom) {
            atomicExch(&grid[n_i], specie); // attacker wins
        } else if (invasion < neighbourDom) {
            atomicExch(&grid[i], neighbour); // defender wins
        }
    }
}