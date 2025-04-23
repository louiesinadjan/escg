#include "config.cuh"

__global__ void refreshRandomNumbers(float* action_probabilities, int* cells, int* neighbours, int numRandomNumbers, int N, bool moore);
__global__ void compute_densities(const int* __restrict__ grid, int* __restrict__ result, int gridSize, int speciesNum);
__global__ void cuda_step(int* grid, float* dominance, int* cells, int* neighbour_dirs, float* invasion_probs, int L); 
__global__ void max_cuda_step(int* grid, float* dominance, int* cells, int* neighbour_dirs, float* invasion_probs, int L, int numRandoms);