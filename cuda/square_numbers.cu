#include <getopt.h>
#include <iostream>

// CUDA Kernel to square each element of an array
__global__ void squareKernel(float* d_out, float* d_in, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main() {
    const int ARRAY_SIZE = 10;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // Host arrays
    float h_in[ARRAY_SIZE], h_out[ARRAY_SIZE];

    // Initialize input array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = i + 1.0f;
    }

    // Device pointers
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    // Copy input data to device
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int blockSize = 256;
    int gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    squareKernel<<<gridSize, blockSize>>>(d_out, d_in, ARRAY_SIZE);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Squared values:\n";
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}