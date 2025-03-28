//
//  main.cpp
//  single-threaded-escg
//
//  Created by Louie Sinadjan on 11/01/2025.
//

#include "config.hpp"
#include "simulation.hpp"
#include "visualise.hpp"

Params parseArgs(int argc, char* argv[]) {
    Params params; // Uses default values

    static struct option long_options[] = {
        {"mcs", required_argument, 0, 'm'},
        {"length", required_argument, 0, 'l'},
        {"height", required_argument, 0, 'h'},
        {"printFrequency", required_argument, 0, 'p'},
        {"neighbourhood", required_argument, 0, 'n'},
        {"species", required_argument, 0, 's'},
        {"mobility", required_argument, 0, 'M'},
        {"flux", required_argument, 0, 'f'},
        {"empty", required_argument, 0, 'w'},
        {"dominance", required_argument, 0, 'd'},
        {"save", required_argument, 0, 's'},
        {"resume", required_argument, 0, 'r'},
        {"numRandoms", required_argument, 0, 'R'},
        {"maxStep", required_argument, 0, 'x'},
        {0, 0, 0, 0} // End of options
    };

    int opt;
    int option_index = 0;

    // Dominance refers to importing a dominance.csv as the dominance adjacency matrix
    std::string usage = "[--mcs <MCS>] [--length <Lattice Length>] [--height <Lattice Height>] "
                        "[--printFrequency <Print Frequency>] [--empty <Initial Empty Cell Probability >] "
                        "[--neighbourhood <Neighbourhood 4/8>] [--mobility <Mobility>] "
                        "[--species <int>] [--flux <true|false>] [--dominance <true|false]"
                        "[--numRandoms <int>][--maxStep <true|false]"
                        "[--save <true|false>] [--resume <true|false>]";

    // Parse the command line arguments
    while ((opt = getopt_long(argc, argv, "m:l:h:p:n:M:s:f:e:r:d:S:x:R:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                params.MCS = std::stoi(optarg);
                break;
            case 'l':
                params.L = std::stoi(optarg);
                break;
            case 'h':
                params.H = std::stoi(optarg);
                break;
            case 'p':
                params.printFrequency = std::stoi(optarg);
                break;
            case 'n':
                params.neighbourhood = std::stoi(optarg);
                if (params.neighbourhood != 4 && params.neighbourhood != 8) {
                    std::cerr << "Error: Neighbourhood must be 4 or 8." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 's':
                params.species = std::stoi(optarg);
                if (params.species < 0) {
                    std::cerr << "Error: Number of species must be >0." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'M':
                params.mobility = std::stof(optarg); // Allows scientific notation input
                break;
            case 'f': {
                std::string fluxStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : fluxStr) {
                    c = tolower(c);
                }
                if (fluxStr == "false" || fluxStr == "0" || fluxStr == "no") {
                    params.flux = false;
                } else {
                    params.flux = true;
                }
                break;
            }
            case 'd': {
                std::string domStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : domStr) {
                    c = tolower(c);
                }
                if (domStr == "true" || domStr == "1" || domStr == "yes") {
                    params.dominance = true;
                } else {
                    params.dominance = false;
                }
                break;
            }
            case 'e':
                params.emptyProbability = std::stof(optarg);
                if (params.emptyProbability <= 0 || params.emptyProbability > 1) {
                    std::cerr << "Error: Initial empty cell probability must be 0 <= p < 1." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'r': {
                std::string resumeStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : resumeStr) {
                    c = tolower(c);
                }
                if (resumeStr == "true" || resumeStr == "1" || resumeStr == "yes") {
                    params.resume = true;
                } else {
                    params.resume = false;
                }
                break;
            }
            case 'S': {
                std::string saveStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : saveStr) {
                    c = tolower(c);
                }
                if (saveStr == "true" || saveStr == "1" || saveStr == "yes") {
                    params.save = true;
                } else {
                    params.save = false;
                }
                break;
            }
            case 'x': {
                std::string maxStepStr(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : maxStepStr) {
                    c = tolower(c);
                }
                if (maxStepStr == "true" || maxStepStr == "1" || maxStepStr == "yes") {
                    params.maxStep = true;
                } else {
                    params.maxStep = false;
                }
                break;
            }
            case 'R':
                params.numRandoms = std::stof(optarg);
                if (params.numRandoms < 1000000) {
                    std::cout << "Entered: " << params.numRandoms << std::endl;
                    std::cerr << "Error: Random numbers must be at least 1,000,000." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                std::cerr << "Usage: " << argv[0] << usage << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    return params;
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

void generateCircularAdjacencyMatrix(int* dominance, int speciesCount) {
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

void initialiseGrid(int* grid, Params p) { // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dist(1, p.species); // Range: 1 to 5 (RPSLS)
    std::uniform_real_distribution<float> emptyProb(0.0f, 1.0f);

    // Randomly initialise the grid with species (stored as int)
    for (int i = 0; i < p.L * p.L; i++) {
        if (emptyProb(gen) < p.emptyProbability) {
            grid[i] = 0; // Randomly assign a species (or EMPTY) as an integer
        } else {
            grid[i] = dist(gen); // Randomly assign a species (or EMPTY) as an integer
        }
    }
}

int main(int argc, const char* argv[]) {
    Params p = parseArgs(argc, const_cast<char**>(argv));
    GridContext gridCtx;
    p.H = p.L;

    int MCS = p.MCS; // 100,000  Monte Carlo Steps

    int L = p.L;          // Length of lattice
    int N = L * L;        // Elementary time steps
    float M = p.mobility; // Mobility 'since it is proportional to the typical area
                          // explored by one mobile individual per unit time'

    int* grid = new int[L * L]; // Grid of L x L

    int* dominance = nullptr; // Dominance matrix
    if (p.dominance) {
        p.species = importCSVToDominance(dominance);
        if (p.species == -1) {
            std::cerr << "Error: Could not import dominance matrix." << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        dominance = new int[p.species * p.species]; // Dominance matrix
        generateCircularAdjacencyMatrix(dominance, p.species);
    }

    float mu = 1;              // RPSLS interaction
    float sigma = 1;           // Reproduction
    float epsilon = 2 * M * N; // Migration

    initialiseGrid(grid, p);

    std::cout << "------------------- Parameters -------------------\n";
    std::cout << "MCS: " << p.MCS << "\n";
    std::cout << "Lattice Length: " << p.L << "\n";
    std::cout << "Initial Empty Cell Probability: " << p.emptyProbability << "\n";
    std::cout << "Neighbourhood: " << p.neighbourhood << "\n";
    std::cout << "Mobility: " << p.mobility << "\n";
    std::cout << "Species: " << p.species << "\n";
    std::cout << "Flux: " << p.flux << "\n";
    std::cout << "Print Frequency: " << p.printFrequency << "\n";
    std::cout << "Save: " << p.save << "\n";
    std::cout << "-------------------------------------------------\n";

    for (int mcs = 0; mcs <= MCS; mcs++) { // Monte Carlo Steps
        densities(gridCtx, grid, p, mcs);  // Every MCS, call densities to add to density vectors for visualisation after simulation

        if (mcs == 0 || mcs == 2000 || mcs == 6000 || mcs == 20000 || mcs == 100000) {
            plot_snapshot(grid, mcs, p);
        }

        for (int n = 0; n < N; n++) { // Elementary Time Steps
            step(p, grid, dominance, mu, sigma, epsilon);
        }
    }

    plot_densities(gridCtx, p);

    std::cout << "Simulation Complete." << std::endl;

    delete[] grid;
    delete[] dominance;
    
    return 0;
}
