#include "io.cuh"
#include <vector>
#include <string>
#include <sstream>

std::vector<std::string> resultsBuffer;

void exportGridToCSV(int* grid, Params p, int mcs) {
    std::ostringstream filename;

    std::string length = "l" + std::to_string(p.L);
    std::string mcs_str = "mcs" + std::to_string(mcs);

    filename << "./" + p.outputDir + "/grid_" << length << "_" << mcs_str << ".csv";

    std::ofstream file(filename.str());
    if (!file) {
        std::cerr << "Error: Could not open file " << filename.str() << " for writing." << std::endl;
        return;
    }

    // Write grid data row-wise
    for (int i = 0; i < p.H; i++) {
        for (int j = 0; j < p.L; j++) {
            file << grid[i * p.L + j]; // Convert 2D index to 1D
            if (j < p.L - 1)
                file << ",";
        }
        file << "\n";
    }

    file << mcs; // Write the MCS at the end of the file

    file.close();
}

// Export the parameters to a CSV file (except bool resume)
void exportParamsToCSV(Params p) {
    std::cout << "Exporting parameters to csv..." << std::endl;

    std::ofstream file("./" + p.outputDir + "/params.csv");
    if (!file) {
        std::cerr << "Error: Could not open file params.csv for writing." << std::endl;
        return;
    }

    file << "MCS," << p.MCS << "\n";
    file << "Lattice Length," << p.L << "\n";
    file << "Alpha," << p.alpha << "\n";
    file << "Beta," << p.beta << "\n";
    file << "Gamma," << p.gamma << "\n";

    file.close();
    std::cout << "Finished exporting parameters to csv" << std::endl;
}

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

void exportDominanceToCSV(float* dominates, int species, Params p) {
    // std::cout << "Exporting dominance to CSV..." << std::endl;

    std::ofstream file("./dominance.csv");
    if (!file) {
        std::cerr << "Error: Could not open file " << p.outputDir << "/dominance.csv for writing." << std::endl;
        return;
    }

    // Write the adjacency matrix row-wise
    for (int i = 0; i < species; i++) {
        for (int j = 0; j < species; j++) {
            file << dominates[i * species + j]; // Convert 1D index to 2D
            if (j < species - 1) {
                file << ","; // Add a comma between values (except last column)
            }
        }
        file << "\n"; // New line after each row
    }

    file.close();
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

void writeResults(int* grid, int L, int mcs, float alpha, float beta, float gamma, int stable) {
    double sp[] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < L * L; i++) {
        sp[grid[i]]++;
    }

    for (int i = 0; i < 8; i++) {
        sp[i] /= (L * L);
    }

    std::ostringstream oss;
    oss << alpha << "," << beta << "," << gamma << "," << mcs << "," << sp[0] << "," << sp[1] << "," << sp[2] << "," << sp[3] << "," 
        << sp[4] << "," << sp[5] << "," << sp[6] << "," << sp[7] << "," << L << "," << stable;
    resultsBuffer.push_back(oss.str());
}

void writeAllResultsToFile(std::string resultsFile) {
    std::ofstream results(resultsFile, std::ios::app);
    if (!results) {
        std::cerr << "Error: Could not open results file for writing." << std::endl;
        return;
    }

    for (const auto& line : resultsBuffer) {
        results << line << "\n";
    }

    results.close();
    resultsBuffer.clear();
}