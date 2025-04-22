#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.hpp"
#include <chrono>
#include <iostream>
#include <random>

void validateRandomNumbers(uint* numbers, int totalNumbers) {
    // Check there are no 0s in the results
    for (int i = 0; i < totalNumbers; i++) {
        if (numbers[i] == 0) {
            std::cerr << "Zero found at index [" << i << "]" << std::endl;
        }
    }
}

void printNumbers(uint* numbers, int totalNumbers) {
    for (int i = 0; i < totalNumbers; i++) {
        std::cout << numbers[i] << std::endl;
    }
}

int main() {
    // Metal Mersenne Twister
    NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Load the Metal library from mt19937.metallib
    NS::Error* error = nullptr;
    NS::String* libraryPath = NS::String::string("build/mt19937.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libraryPath, &error);
    if (!library) {
        std::cerr << "Error: Failed to load mt19937.metallib - " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    // Load the mersenne_twister function
    MTL::Function* mersenne_twister = library->newFunction(NS::String::string("mersenne_twister", NS::UTF8StringEncoding));
    if (!mersenne_twister) {
        std::cerr << "Error: Failed to find the function 'mersenne_twister' in mt19937.metallib." << std::endl;
        return -1;
    }

    // Create a compute pipeline state
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(mersenne_twister, &error);
    if (!pipelineState) {
        std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    // Prepare buffers
    const int numThreads = 2000;
    const int numRandomNumbers = 1'000'000'000;

    uint32_t seeds[numThreads];

    for (int i = 0; i < numThreads; ++i) {
        seeds[i] = i + 1;
    }

    MTL::Buffer* seedBuffer = device->newBuffer(seeds, sizeof(seeds), MTL::ResourceStorageModeShared);
    uint32_t* results = new uint32_t[numRandomNumbers];
    MTL::Buffer* resultBuffer = device->newBuffer(sizeof(uint32_t) * numRandomNumbers, MTL::ResourceStorageModeShared);

    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState); //
    encoder->setBuffer(seedBuffer, 0, 0);
    encoder->setBuffer(resultBuffer, 0, 1);

    // Dispatch threads
    MTL::Size gridSize = MTL::Size(numThreads, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(pipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();

    // Commit command buffer and wait for completion
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Retrieve results
    results = static_cast<uint32_t*>(resultBuffer->contents());

    // validateRandomNumbers(results, numRandomNumbers);

    // printNumbers(results, numRandomNumbers);

    autoreleasePool->release();

    return 0;
}