#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void curand_numbers(uint* results, int numPerThread) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (gridDim.x * blockDim.x))
        return; // Prevent out-of-bounds access

    // Initialise cuRAND
    curandState state;
    curand_init(clock64(), id, 0, &state);

    // Generate random numbers
    for (int i = 0; i < numPerThread; i++) {
        results[id * numPerThread + i] = curand(&state);
    }
}

void generateRandomNumbers(int totalNumbers, uint* h_results) {
    int threadsPerBlock = 256;
    int numThreads = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    int numPerThread = (totalNumbers + (blocksPerGrid * threadsPerBlock) - 1) / (blocksPerGrid * threadsPerBlock);

    if (numPerThread == 0) {
        numPerThread = 1; // Ensure at least one number per thread
    }

    // Allocate memory for seeds and results
    uint* d_results;

    if (cudaMalloc(&d_results, totalNumbers * sizeof(uint)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_results" << std::endl;
        return;
    }

    // Copy seeds to device
    cudaMemset(d_results, 0, totalNumbers * sizeof(uint));

    // Launch Kernel
    curand_numbers<<<blocksPerGrid, threadsPerBlock>>>(d_results, numPerThread);
    cudaDeviceSynchronize(); // Ensure all threads complete before copying

    // Copy results back to host
    cudaMemcpy(h_results, d_results, totalNumbers * sizeof(uint), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_results);
}

void validateRandomNumbers(uint* h_results, int totalNumbers) {
    // Check there are no 0s in the results
    for (int i = 0; i < totalNumbers; i++) {
        if (h_results[i] == 0) {
            std::cerr << "Zero found at index [" << i << "]" << std::endl;
        }
    }
}

void printNumbers(uint* h_results, int totalNumbers) {
    for (int i = 0; i < totalNumbers; i++) {
        std::cout << h_results[i] << std::endl;
    }
}

int main() {
    int totalNumbers = 10'000'000; 
    uint* h_results = new uint[totalNumbers];
    memset(h_results, 0, totalNumbers * sizeof(uint));

    for(int i = 0; i < 100; i++) {
        generateRandomNumbers(totalNumbers, h_results);
    }

    // validateRandomNumbers(h_results, totalNumbers);
    // printNumbers(h_results, totalNumbers);

    delete[] h_results;
    return 0;
}