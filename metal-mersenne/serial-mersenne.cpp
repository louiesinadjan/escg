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
    static std::random_device rd;                     // Random number generator
    static std::mt19937 gen(rd());                    // Mersenne Twister
    std::uniform_int_distribution<int> dist(INT_MAX); // Uniform distribution

    int* x = new int[1000000]; // Array to store random numbers
    for (int run = 0; run < 100; run++) {
        for (int j = 0; j < 1000000; j++) { // Generate 1,000,000 random numbers
            x[j] = dist(gen);
        }
    }

    // validateRandomNumbers(x, 1000000);
    // printNumbers(x, 1000000);

    delete[] x;
    return 0;
}