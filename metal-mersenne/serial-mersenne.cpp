#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

void validateRandomNumbers(int* numbers, int totalNumbers) {
    // Check there are no 0s in the results
    for (int i = 0; i < totalNumbers; i++) {
        if (numbers[i] == 0) {
            std::cerr << "Zero found at index [" << i << "]" << std::endl;
        }
    }
}

void printNumbers(int* numbers, int totalNumbers) {
    for (int i = 0; i < totalNumbers; i++) {
        std::cout << numbers[i] << std::endl;
    }
}

int main() {
    // Serial Mersenne Twister
    static std::random_device rd;                        // Random number generator
    static std::mt19937 gen(rd());                       // Mersenne Twister
    std::uniform_int_distribution<int> dist(1, INT_MAX); // Uniform distribution

    int numRandoms = 100'000'000; // Number of random numbers to generate

    int* x = new int[numRandoms]; // Array to store random numbers
    int* target = new int[numRandoms]; // Array to store target numbers

    for (int i = 0; i < 10; i++) { // Generate 1,000,000,000 random numbers
        for (int j = 0; j < numRandoms; j++) {
            x[j] = dist(gen);
        }
        std::memcpy(target, x, numRandoms * sizeof(int)); // Copy the generated numbers to target
    }


    // validateRandomNumbers(x, 1000000);
    // printNumbers(x, 1000000);

    delete[] x;
    return 0;
}