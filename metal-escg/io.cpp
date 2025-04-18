#include "io.hpp"

void exportGridToCSV(int* grid, Params p, int mcs) {
    std::ostringstream filename;

    bool moore = p.neighbourhood == 8;
    std::string length = "l" + std::to_string(p.L);
    std::string height = "h" + std::to_string(p.H);
    std::string neighbourhood = moore ? "Moore" : "VN";
    std::string mcs_str = "mcs" + std::to_string(mcs);

    // Convert to scientific notation
    std::ostringstream oss;
    oss.precision(2);
    oss << std::scientific << p.mobility;
    std::string mobility_str = "M" + oss.str();
    std::string flux = p.flux ? "flux" : "noflux";
    std::string species = std::to_string(p.species) + "species";

    filename << "./" + p.outputDir + "/grid_" << length << "_" << height << "_" << neighbourhood << "_" << mobility_str << "_" << flux << "_" << species << "_" << mcs_str << ".csv";

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
    file << "Lattice Height," << p.H << "\n";
    file << "Initial Empty Cell Probability," << p.emptyProbability << "\n";
    file << "Neighbourhood," << p.neighbourhood << "\n";
    file << "Mobility," << p.mobility << "\n";
    file << "Species," << p.species << "\n";
    file << "Flux," << p.flux << "\n";
    file << "Print Frequency," << p.printFrequency << "\n";
    file << "Number of Randoms," << p.numRandoms << "\n";
    file << "Max Step," << p.maxStep << "\n";
    

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

void importCSVToParams(Params& p) { // Imports the parameters (except resume) from the csv file
    std::ifstream file("params.csv");
    if (!file) {
        std::cerr << "Error: Could not open file params.csv for reading." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key, value;
        std::getline(ss, key, ',');
        std::getline(ss, value, ',');
        if (key == "MCS") {
            p.MCS = std::stoi(value);
        } else if (key == "Lattice Length") {
            p.L = std::stoi(value);
        } else if (key == "Lattice Height") {
            p.H = std::stoi(value);
        } else if (key == "Initial Empty Cell Probability") {
            p.emptyProbability = std::stof(value);
        } else if (key == "Neighbourhood") {
            p.neighbourhood = std::stoi(value);
        } else if (key == "Mobility") {
            p.mobility = std::stof(value);
        } else if (key == "Species") {
            p.species = std::stoi(value);
        } else if (key == "Flux") {
            p.flux = value == "true" ? true : false;
        } else if (key == "Print Frequency") {
            p.printFrequency = std::stoi(value);
        }
    }

    file.close();
}

int importCSVToDominance(int*& dominates) { // `dominates` passed by reference
    std::ifstream file("dominance.csv");
    if (!file) {
        std::cerr << "Error: Could not open file dominance.csv for reading." << std::endl;
        return -1; // Return error code
    }

    std::vector<std::vector<int>> matrix; // Temporary 2D matrix
    std::string line;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<int> rowValues;

        while (std::getline(ss, cell, ',')) {
            rowValues.push_back(std::stoi(cell)); // Convert CSV values to integers
        }

        matrix.push_back(rowValues);
    }

    file.close();

    // Get species count (assume square matrix)
    int species = matrix.size();

    // Allocate memory for `dominates` (caller must free it)
    dominates = new int[species * species];

    // Flatten the 2D matrix into the 1D array
    for (int i = 0; i < species; i++) {
        for (int j = 0; j < species; j++) {
            dominates[i * species + j] = matrix[i][j]; // Row-major flattening
        }
    }

    return species; // Return species count
}

void exportDominanceToCSV(int* dominates, int species, Params p) {
    std::cout << "Exporting dominance to CSV..." << std::endl;

    std::ofstream file("./" + p.outputDir + "/dominance.csv");
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

void generateCirculantAdjacencyMatrix(int* dominance, int speciesCount) {
    // Choose default offsets based on species count
    std::vector<int> offsets = (speciesCount >= 5) ? std::vector<int>{1, 3} : std::vector<int>{1};
    if (speciesCount < 2) {
        offsets = {};
    } else if (speciesCount < 5) {
        offsets = {1};
    } else if (speciesCount >= 5 && speciesCount < 8) {
        offsets = {1, 3};
    } else {
        offsets = {1, 3, 5, 7};
    }

    // Initialise dominance matrix (1D representation of speciesCount x speciesCount)
    for (int i = 0; i < speciesCount * speciesCount; i++) {
        dominance[i] = 0; // Set everything to 0
    }

    // Fill the adjacency matrix
    for (int species = 0; species < speciesCount; species++) {
        for (int offset : offsets) {
            int target = (species + offset) % speciesCount; // Circular wrap-around
            dominance[species * speciesCount + target] = 1; // Row-major storage
        }
    }
}