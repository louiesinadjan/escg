#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.hpp"
#include "timer.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

int main(int argc, char* argv[]) {
    // Serial Mersenne Twister
    static std::random_device rd;                        // Random number generator
    static std::mt19937 gen(rd());                       // Mersenne Twister
    std::uniform_int_distribution<int> dist(4294967295); // Uniform distribution

    double serialTimes[100];
    for (int run = 0; run < 100; ++run) {
        Timer timer;
        timer.start();                      // Start the timer for serial Mersenne Twister
        int x[1000000];                     // Array to store random numbers
        for (int j = 0; j < 1000000; j++) { // Generate 1,000,000 random numbers
            x[j] = dist(gen);
        }
        timer.stop(); // Stop the timer
        serialTimes[run] = timer.elapsedMilliseconds();
    }
    double avgSerialTime = std::accumulate(serialTimes, serialTimes + 100, 0.0) / 100.0;
    std::cout << "Serial Completed" << std::endl;

    // Metal Mersenne Twister
    NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Load the Metal library from mt19937.metallib
    NS::Error* error = nullptr;
    NS::String* libraryPath = NS::String::string("/Users/louiesinadjan/Documents/dissertation/escg/metal-mersenne/build/mt19937.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libraryPath, &error);
    if (!library) {
        std::cerr << "Error: Failed to load mt19937.metallib - " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }
    std::cout << "Metal Library Loaded" << std::endl;

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
    const int numRandomNumbers = 1000000;
    uint32_t seeds[numThreads];
    for (int i = 0; i < numThreads; ++i) {
        seeds[i] = i + 1;
    }

    MTL::Buffer* seedBuffer = device->newBuffer(seeds, sizeof(seeds), MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBuffer = device->newBuffer(sizeof(uint32_t) * numRandomNumbers, MTL::ResourceStorageModeShared);

    double metalTimes[100];
    for (int run = 0; run < 100; ++run) {
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

        // Start timer
        Timer t2;
        t2.start();

        // Commit command buffer and wait for completion
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        // Stop timer
        t2.stop();
        metalTimes[run] = t2.elapsedMilliseconds();

        // Retrieve results
        uint32_t* results = static_cast<uint32_t*>(resultBuffer->contents());

        // Verify the number of generated random numbers
        bool valid = true;
        for (int i = 0; i < numRandomNumbers; ++i) {
            if (results[i] == 0) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            std::cerr << "Error: Not all random numbers were generated correctly in run " << run << std::endl;
        }
    }
    double avgMetalTime = std::accumulate(metalTimes, metalTimes + 100, 0.0) / 100.0;

    for (int run = 0; run < 100; ++run) {
        std::cout << run << ": Serial = " << serialTimes[run] << " ms, Metal = " << metalTimes[run] << std::endl;
    }

    std::cout << "Average Serial Mersenne Twister: " << avgSerialTime << " ms" << std::endl;
    std::cout << "Average Metal Mersenne Twister: " << avgMetalTime << " ms" << std::endl;

    // Clean up
    autoreleasePool->release();

    return 0;
}