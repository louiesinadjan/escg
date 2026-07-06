#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Chi-squared statistic for uniformity of `numbers` over `numBins` equal-width bins.
// numBins = 256 divides 2^32 exactly, so binning on numbers[i] % numBins introduces no modulo bias.
double chiSquaredStatistic(const uint32_t* numbers, size_t n, int numBins) {
    std::vector<uint64_t> counts(numBins, 0);
    for (size_t i = 0; i < n; ++i) {
        counts[numbers[i] % numBins]++;
    }
    double expected = static_cast<double>(n) / numBins;
    double chi2 = 0.0;
    for (int b = 0; b < numBins; ++b) {
        double diff = static_cast<double>(counts[b]) - expected;
        chi2 += diff * diff / expected;
    }
    return chi2;
}

// Wilson-Hilferty approximation: converts a chi-squared statistic to a
// standard-normal z-score so we get a p-value without a stats library.
double chiSquaredPValue(double chi2, int dof) {
    double t1 = std::cbrt(chi2 / dof);
    double t2 = 1.0 - 2.0 / (9.0 * dof);
    double t3 = std::sqrt(2.0 / (9.0 * dof));
    double z = (t1 - t2) / t3;
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

// Serial test: chi-squared goodness-of-fit on consecutive pairs (numbers[i], numbers[i+1])
// binned into a binsPerDim x binsPerDim grid. Tests independence between successive outputs
// (uniformity alone can't catch a generator whose outputs are correlated).
double serialTestChiSquared(const uint32_t* numbers, size_t n, int binsPerDim) {
    size_t numCells = static_cast<size_t>(binsPerDim) * binsPerDim;
    std::vector<uint64_t> counts(numCells, 0);
    for (size_t i = 0; i + 1 < n; ++i) {
        uint32_t a = numbers[i] % binsPerDim;
        uint32_t b = numbers[i + 1] % binsPerDim;
        counts[a * binsPerDim + b]++;
    }
    double expected = static_cast<double>(n - 1) / numCells;
    double chi2 = 0.0;
    for (size_t c = 0; c < numCells; ++c) {
        double diff = static_cast<double>(counts[c]) - expected;
        chi2 += diff * diff / expected;
    }
    return chi2;
}

// Lag-1 autocorrelation coefficient. For an IID sequence this should be near 0;
// the approximate 95% significance bound under the null is +/- 1.96/sqrt(n).
double lag1Autocorrelation(const uint32_t* numbers, size_t n) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += static_cast<double>(numbers[i]);
    mean /= n;

    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(numbers[i]) - mean;
        denominator += d * d;
        if (i + 1 < n) {
            double dNext = static_cast<double>(numbers[i + 1]) - mean;
            numerator += d * dNext;
        }
    }
    return numerator / denominator;
}

void validateRandomNumbers(uint32_t* numbers, int totalNumbers, int numThreads, int numBins = 256, double alpha = 0.05) {
    // 1. Frequency (uniformity) test
    double chi2 = chiSquaredStatistic(numbers, totalNumbers, numBins);
    int dof = numBins - 1;
    double p = chiSquaredPValue(chi2, dof);

    // 2. Serial (independence) test - 16x16 grid keeps expected count per cell large
    int binsPerDim = 16;
    double serialChi2 = serialTestChiSquared(numbers, totalNumbers, binsPerDim);
    int serialDof = binsPerDim * binsPerDim - 1;
    double serialP = chiSquaredPValue(serialChi2, serialDof);

    // 3. Lag-1 autocorrelation
    double autocorr = lag1Autocorrelation(numbers, totalNumbers);
    double autocorrBound = 1.96 / std::sqrt(static_cast<double>(totalNumbers));

    std::cout << "numbers generated = " << totalNumbers << ", parallel threads = " << numThreads << "\n"
              << "Frequency test:  chi2 = " << chi2 << ", dof = " << dof << ", p-value = " << p
              << "  [" << (p > alpha ? "PASS" : "FAIL") << "]\n"
              << "Serial test:     chi2 = " << serialChi2 << ", dof = " << serialDof << ", p-value = " << serialP
              << "  [" << (serialP > alpha ? "PASS" : "FAIL") << "]\n"
              << "Lag-1 autocorr:  r = " << autocorr << ", bound = +/-" << autocorrBound
              << "  [" << (std::fabs(autocorr) < autocorrBound ? "PASS" : "FAIL") << "]"
              << std::endl;
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

    for (int trial = 0; trial < 10; trial++) { // Generate 1,000,000,000 random numbers
        // Prepare buffers
        const int numThreads = 2000;
        const int numRandomNumbers = 100'000'000;
        int* randoms = new int[numRandomNumbers];

        uint32_t seeds[numThreads];

        for (int t = 0; t < numThreads; ++t) {
            seeds[t] = trial * numThreads + t + 1; // distinct seeds per trial so trials are IID
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
        std::memcpy(randoms, resultBuffer->contents(), sizeof(uint32_t) * numRandomNumbers);

        validateRandomNumbers(reinterpret_cast<uint32_t*>(randoms), numRandomNumbers, numThreads);
    }

    autoreleasePool->release();

    return 0;
}