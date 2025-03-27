//
//  main.cpp
//  single-threaded-escg
//
//  Created by Louie Sinadjan on 11/01/2025.
//

#include "config.hpp"
#include "io.hpp"
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
        {"save", required_argument, 0, 'S'},
        {"resume", required_argument, 0, 'r'},
        {"numRandoms", required_argument, 0, 'R'},
        {"maxStep", required_argument, 0, 'x'},

        {"alpha", required_argument, 0, 'a'},
        {"beta", required_argument, 0, 'b'},
        {"gamma", required_argument, 0, 'g'},

        {0, 0, 0, 0} // End of options
    };

    int opt;
    int option_index = 0;

    // Dominance refers to importing a dominance.csv as the dominance adjacency matrix
    std::string usage = "[--mcs <MCS>] [--length <Lattice Length>] [--height <Lattice Height>] "
                        "[--printFrequency <Print Frequency>] [--empty <Initial Empty Cell Probability >] "
                        "[--neighbourhood <Neighbourhood 4/8>] [--mobility <float>] "
                        "[--species <int>] [--flux <true|false>] [--dominance <true|false]"
                        "[--numRandoms <int>][--maxStep <true|false]"
                        "[--save <true|false>] [--resume <true|false>]"
                        "[--alpha <float>] [--beta <float>] [--gamma <float>]";

    // Parse the command line arguments
    while ((opt = getopt_long(argc, argv, "m:l:h:p:n:M:s:f:e:r:d:S:x:R:a:b:g:", long_options, &option_index)) != -1) {
        switch (opt) {
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
            case 'e':
                params.emptyProbability = std::stof(optarg);
                if (params.emptyProbability <= 0 || params.emptyProbability > 1) {
                    std::cerr << "Error: Initial empty cell probability must be 0 <= p < 1." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
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
            case 'a':
                params.alpha = std::stof(optarg);
                break;
            case 'b':
                params.beta = std::stof(optarg);
                break;
            case 'g':
                params.gamma = std::stof(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << usage << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    return params;
}

void initialiseGrid(int* grid, Params p) { // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dist(0, 7); // Range: 1 to 5 (RPSLS)
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

bool compareGrid(int* grid, int* prevGrid, int L, int mcs, Params p) {
    bool same = std::memcmp(grid, prevGrid, sizeof(int) * L * L) == 0;

    if (same) {
        if (mcs < 10000) {
            writeResults(grid, L, 10000, p.alpha, p.beta, true);
            writeResults(grid, L, 50000, p.alpha, p.beta, true);
            writeResults(grid, L, 100000, p.alpha, p.beta, true);
        } else if (mcs < 50000) {
            writeResults(grid, L, 50000, p.alpha, p.beta, true);
            writeResults(grid, L, 100000, p.alpha, p.beta, true);
        } else if (mcs < 100000) {
            writeResults(grid, L, 100000, p.alpha, p.beta, true);
        }
    }

    return same;
}

int main(int argc, const char* argv[]) {
    Params p = parseArgs(argc, const_cast<char**>(argv));
    GridContext gridCtx;
    p.H = p.L;
    p.species = 8;

    int MCS = p.MCS; // 100,000  Monte Carlo Steps

    int L = p.L;   // Length of lattice
    int N = L * L; // Elementary time steps

    int* grid = new int[N]; // Grid of L x L
    int* prevGrid = new int[N];
    std::fill(prevGrid, prevGrid + N, -1);

    float* dominance = new float[64]; // Dominance matrix

    generateDominance(dominance, p.alpha, p.beta, p.gamma);

    initialiseGrid(grid, p);
    if (p.resume) {
        importCSVToGrid(grid, N);
    }

    for (int mcs = 0; mcs <= MCS; mcs++) {            // Monte Carlo Steps
        if (compareGrid(grid, prevGrid, L, mcs, p)) { // If grid same as last step, break
            std::cout << "Stasis reached at MCS: " << mcs << std::endl;
            break;
        }

        densities(gridCtx, grid, p, mcs); // Every MCS, call densities to add to density vectors for visualisation after simulation

        if (mcs <= 10) {
            writeResults(grid, L, mcs, p.alpha, p.beta, false);
            if (p.save) {
                plot_snapshot(grid, mcs, p);
            }
        } else if (mcs == 50 || mcs == 100 || mcs == 1000 || mcs == 5000) {
            writeResults(grid, L, mcs, p.alpha, p.beta, false);
            if (p.save) {
                plot_snapshot(grid, mcs, p);
            }
        } else if (mcs % 10000 == 0) {
            writeResults(grid, L, mcs, p.alpha, p.beta, false);
            if (p.save) {
                plot_snapshot(grid, mcs, p);
            }
        }
        std::memcpy(prevGrid, grid, sizeof(int) * N);
        for (int n = 0; n < N; n++) { // Elementary Time Steps
            step(p, grid, dominance);
        }
    }

    plot_densities(gridCtx, p);

    std::cout << "Simulation Complete." << std::endl;

    delete[] grid;
    delete[] prevGrid;
    delete[] dominance;

    return 0;
}
