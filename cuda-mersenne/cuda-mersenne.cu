#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define MT_N 624
#define MT_M 397
#define MATRIX_A 0x9908b0df
#define UPPER_MASK 0x80000000
#define LOWER_MASK 0x7fffffff

// Mersenne Twister structure
struct MT19937 {
    uint state[MT_N];
    uint index;
};

// CUDA Device Function: Initialise Mersenne Twister with Seed
__device__ void initialise(MT19937& mt, uint seed) {
    mt.state[0] = seed;
    for (uint i = 1; i < MT_N; ++i) {
        mt.state[i] = 1812433253 * (mt.state[i - 1] ^ (mt.state[i - 1] >> 30)) + i;
    }
    mt.index = MT_N;
}

// CUDA Device Function: Perform Twisting Transformation
__device__ void twist(MT19937& mt) {
    for (uint i = 0; i < MT_N; ++i) {
        uint y = (mt.state[i] & UPPER_MASK) + (mt.state[(i + 1) % MT_N] & LOWER_MASK);
        mt.state[i] = mt.state[(i + MT_M) % MT_N] ^ (y >> 1);
        if (y % 2 != 0) {
            mt.state[i] ^= MATRIX_A;
        }
    }
    mt.index = 0;
}

// CUDA Device Function: Extract Random Number
__device__ uint extract(MT19937& mt) {
    if (mt.index >= MT_N) {
        twist(mt);
    }
    uint y = mt.state[mt.index++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    return y;
}

// CUDA Kernel: Generate Random Numbers Using Mersenne Twister
__global__ void mersenne_twister(const uint* seeds, uint* results, int numPerThread) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (gridDim.x * blockDim.x))
        return; // Prevent out-of-bounds access

    // Each thread gets a unique seed from the array
    uint thread_seed = seeds[id];

    // Each thread gets its own Mersenne Twister state
    MT19937 mt;
    initialise(mt, thread_seed);

    // Generate `numPerThread` random numbers
    for (int i = 0; i < numPerThread; ++i) {
        results[id * numPerThread + i] = extract(mt);
    }
}

// Host Function: Allocate Memory & Launch Kernel
void generateRandomNumbers(int totalNumbers, uint* h_results) {
    int threadsPerBlock = 256;
    int numThreads = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    int numPerThread = (totalNumbers + (blocksPerGrid * threadsPerBlock) - 1) / (blocksPerGrid * threadsPerBlock);

    if (numPerThread == 0) {
        numPerThread = 1; // Ensure at least one number per thread
    }

    // Allocate memory for seeds and results
    uint* h_seeds = new uint[numThreads];
    uint *d_seeds, *d_results;

    if (cudaMalloc(&d_seeds, numThreads * sizeof(uint)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_seeds" << std::endl;
        return;
    }
    if (cudaMalloc(&d_results, totalNumbers * sizeof(uint)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_results" << std::endl;
        return;
    }

    // Generate time-based seed
    uint base_seed = static_cast<uint>(std::time(0));

    // Assign each thread a unique seed
    for (int i = 0; i < numThreads; i++) {
        h_seeds[i] = base_seed + i; // Ensure different seeds for different threads
    }

    // Copy seeds to device
    cudaMemcpy(d_seeds, h_seeds, numThreads * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, totalNumbers * sizeof(uint));

    // Launch Kernel
    mersenne_twister<<<blocksPerGrid, threadsPerBlock>>>(d_seeds, d_results, numPerThread);
    cudaDeviceSynchronize(); // Ensure all threads complete before copying

    // Copy results back to host
    cudaMemcpy(h_results, d_results, totalNumbers * sizeof(uint), cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_seeds;
    cudaFree(d_seeds);
    cudaFree(d_results);
}

// Host Function: Validate Random Numbers
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

// Main Function
int main() {
    int totalNumbers = 1000000; // Generate exactly 1,000,000 random numbers
    uint* h_results = new uint[totalNumbers];
    memset(h_results, 0, totalNumbers * sizeof(uint));

    for (int i = 0; i < 100; i++) {
        generateRandomNumbers(totalNumbers, h_results);
    }

    // validateRandomNumbers(h_results, totalNumbers);
    // printNumbers(h_results, totalNumbers);

    delete[] h_results;
    return 0;
}