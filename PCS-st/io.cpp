#include "io.hpp"

int importCSVToGrid(int* grid, int N) {
    std::ifstream file("grid.csv");
    if (!file) {
        std::cerr << "Error: Could not open file grid.csv for reading." << std::endl;
        return -1;
    }

    std::string line;
    int i = 0;
    std::string lastLine; // Store the last line separately

    while (std::getline(file, line)) {
        if (!file.eof()) { // Read grid values normally
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',') && i < N) {
                grid[i] = std::stoi(cell);
                i++;
            }
        }
        lastLine = line; // Store the last read line as MCS
    }

    file.close();

    try {
        return std::stoi(lastLine);
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to convert last line to MCS: " << lastLine << std::endl;
        return -1;
    }
}

void generateDominance(float* dominance, float alpha, float beta, float gamma) {
    // Initialise the dominance matrix to zero
    std::fill(dominance, dominance + 64, 0.0f); // Assuming 8 species, 8x8 matrix

    // Fill the adjacency matrix with dominance rules
    for (int species = 0; species < 8; species++) {
        // Cyclic dominance
        dominance[species * 8 + (species + 1) % 8] = gamma; // Species i invades (i+1) mod 8
        // Secondary dominance
        dominance[species * 8 + (species + 2) % 8] = alpha; // Species i invades (i+2) mod 8
    }

    // Asymmetrical dominance
    dominance[2 * 8 + 6] = beta; // Species 2 invades species 6
    dominance[4 * 8 + 0] = beta; // Species 4 invades species 0
}

void writeResults(int* grid, int L, int mcs, float alpha, float beta, bool stable) {
    double sp[] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < L * L; i++) {
        sp[grid[i]]++;
    }

    for (int i = 0; i < 8; i++) {
        sp[i] /= (L * L);
    }

    std::ofstream results;
    results.open("results2.csv", std::ios::app);
    results << alpha << "," << beta << "," << mcs << "," << sp[0] << "," << sp[1] << "," << sp[2] << "," << sp[3] << "," << sp[4] << "," << sp[5] << "," << sp[6] << "," << sp[7] << "," << L << ","
            << (stable ? "true" : "false") << "\n";
    results.close();
}