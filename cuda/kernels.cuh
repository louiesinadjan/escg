#include "config.cuh"

__global__ void refreshRandomNumbers(float* action_probabilities, int* cells, int* neighbours, int numRandomNumbers, int N, bool moore);
__global__ void compute_densities(const int* __restrict__ grid, int* __restrict__ result, int gridSize, int speciesNum);
__global__ void cuda_step(int* grid, int* dominance, int* cells, int* neighbour_dirs, float* action_probs, float mu, float sigma, int L, int H, int speciesNum);
__global__ void max_cuda_step(int* grid, int* dominance, int* cells, int* neighbour_dirs, float* action_probs, float mu, float sigma, int L, int H, int speciesNum, int numRandoms);
