#include "config.cuh"
#include "io.cuh"
#include "kernels.cuh"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <vector>
#include <cstring>

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

        {"resultsFile", required_argument, 0, 'o'},

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
                        "[--alpha <float>] [--beta <float>] [--gamma <float>]"
                        "[--resultsFile <file.csv>]";

    // Parse the command line arguments
    while ((opt = getopt_long(argc, argv, "m:l:h:p:n:s:f:e:r:d:S:x:R:a:b:g:o:", long_options, &option_index)) != -1) {
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
            case 'p': {
                std::string patchwork(optarg);
                // Convert the argument to lowercase for easier comparison
                for (auto& c : patchwork) {
                    c = tolower(c);
                }
                if (patchwork == "true" || patchwork == "1" || patchwork == "yes") {
                    params.patchwork = true;
                } else {
                    params.patchwork = false;
                }
                break;
            }
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
                if (maxStepStr == "false" || maxStepStr == "0" || maxStepStr == "no") {
                    params.maxStep = false;
                } else {
                    params.maxStep = true;
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
            case 'a':
                params.alpha = std::stof(optarg);
                break;
            case 'b':
                params.beta = std::stof(optarg);
                break;
            case 'g':
                params.gamma = std::stof(optarg);
                break;
            case 'o': {
                std::string resultsFile(optarg);
                params.resultsFile = resultsFile;
                break;
            }
            default:
                std::cerr << "Usage: " << argv[0] << usage << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    return params;
}

void generateRandomNumbers(float* d_invasion_probabilities, int* d_cells, int* d_neighbours, int N, int numRandomNumbers, bool moore, cudaStream_t stream_numbers) {
    // std::cout << "\nRefreshing random numbers..." << std::endl;
    // std::cout << std::endl;

    int minGridSize = 0, bestBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, refreshRandomNumbers, 0, 0);

    // Adjust the block size to fit within totalThreads
    int totalThreads = numRandomNumbers / 10000;
    int threadsPerBlock = std::min(bestBlockSize, totalThreads);
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    refreshRandomNumbers<<<blocksPerGrid, threadsPerBlock, 0, stream_numbers>>>(d_invasion_probabilities, d_cells, d_neighbours, numRandomNumbers, N, moore);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (refreshRandomNumbers): " << cudaGetErrorString(err) << std::endl;
    }
}

void initialiseGrid(int* h_grid, Params p) {
    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 7);

    // Randomly initialise the grid with species (stored as int)
    for (int i = 0; i < p.L * p.H; i++) {
        h_grid[i] = dist(gen);
    }
}

bool compareGrid(int* grid, int* prev, int L, int mcs, Params p) {
    bool same = std::memcmp(grid, prev, sizeof(int) * L * L) == 0;

    if (same) {
        std::vector<int> thresholds;
        // int step = p.numRandoms / (L * L);
        // step = step * 10;
        // for (int t = step; t <= 500000; t += step) {
        //     thresholds.push_back(t);
        // }
        // int upper = ((500000 + step - 1) / step) * step;
        // if (thresholds.empty() || thresholds.back() < upper) {
        //     thresholds.push_back(upper);
        // }
        // thresholds = {1000, 2000, 3000, 5000, 7000, 10000, 25000, 50000, 100000, 250000, 500000};

        // thresholds = {1000, 2000, 3000, 5000, 7000, 10000, 25000, 50000, 160000};
        thresholds = {1000, 2000, 3000, 5000, 7000, 10000, 25000, 50000};

        for (int t : thresholds) {
            if (mcs < t)
                writeResults(grid, L, t, p.alpha, p.beta, p.gamma, mcs);
        }
    }

    return same;
}

void densities(int* grid, int* d_grid, int* d_speciesCounts, int N, int mcs, int printFrequency, GridContext& gridCtx, int speciesNum, std::set<int>& speciesSet) {
    cudaError_t err = cudaMemcpy(d_grid, grid, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_grid: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaMemset(d_speciesCounts, 0, sizeof(int) * (speciesNum));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    compute_densities<<<blocksPerGrid, threadsPerBlock>>>(d_grid, d_speciesCounts, N, speciesNum);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (compute_densities): " << cudaGetErrorString(err) << std ::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (compute_densities): " << cudaGetErrorString(err) << std ::endl;
    }

    // Copy results back to host
    int* speciesCounts = new int[speciesNum]; // Species
    cudaMemcpy(speciesCounts, d_speciesCounts, sizeof(int) * (speciesNum), cudaMemcpyDeviceToHost);

    std::vector<double> densities;
    for (int i = 0; i <= 7; i++) {
        densities.push_back((static_cast<double>(speciesCounts[i]) / N) * 100);

        if (speciesCounts[i] == 0) {
            speciesSet.erase(i);
        }
    }

    gridCtx.steps.push_back(mcs);
    gridCtx.speciesDensities.push_back(densities);

    // Print the densities
    if (mcs % printFrequency == 0) {
        std::cout << "Population Densities at: " << mcs << std::endl;

        for (int i = 0; i < speciesNum; i++) {
            // The vector is 0-indexed but the species starts from Species 1 (value 0 in grid refers to empty)
            std::cout << "Species " << (i) << ": " << densities[i] << std::endl;
        }
        std::cout << std::endl;
    }

    delete[] speciesCounts;
}

void maxStep(int* h_grid, int* d_grid, float* d_dominance, int* d_cells, int* d_neighbour_dirs, float* d_invasion_probabilities, Params& p, cudaStream_t stream_step) {
    int N = p.L * p.H;

    cudaMemcpy(d_grid, h_grid, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (p.numRandoms + threadsPerBlock - 1) / threadsPerBlock;
    max_cuda_step<<<blocksPerGrid, threadsPerBlock, 0, stream_step>>>(d_grid, d_dominance, d_cells, d_neighbour_dirs, d_invasion_probabilities, p.L, p.numRandoms);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (cuda_step): " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_grid, d_grid, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream_step);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (cuda_step): " << cudaGetErrorString(err) << std ::endl;
    }
}

void step(int* h_grid, int* d_grid, float* d_dominance, int* d_cells, int* d_neighbour_dirs, float* d_invasion_probabilities, Params& p, cudaStream_t stream_step) {
    int N = p.L * p.H;

    cudaMemcpy(d_grid, h_grid, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cuda_step<<<blocksPerGrid, threadsPerBlock, 0, stream_step>>>(d_grid, d_dominance, d_cells, d_neighbour_dirs, d_invasion_probabilities, p.L);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (cuda_step): " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_grid, d_grid, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream_step);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (cuda_step): " << cudaGetErrorString(err) << std ::endl;
    }
}

bool stasis(std::set<int> speciesSet) { return speciesSet.size() <= 1; }

int main(int argc, const char* argv[]) {
    // ------------------- Parse Command Line Arguments -------------------
    Params params = parseArgs(argc, const_cast<char**>(argv));

    int MCS = params.MCS;
    int L;       // Length of lattice
    int N;       // Elementary time steps = total number of cells
    bool moore;  // Moore neighbourhood if true, Von Neumann if false
    int* h_grid; // Flattened grid size = L x H

    L = params.L;
    params.H = params.L; // Everything is square
    N = L * L;
    MCS = params.MCS;
    moore = params.neighbourhood == 8;
    h_grid = new int[N];

    // Creating output directory
    std::string length = "l" + std::to_string(L);
    std::string alpha = "alpha" + std::to_string(params.alpha);
    std::string beta = "beta" + std::to_string(params.beta);
    std::string gamma = "gamma" + std::to_string(params.gamma);
    params.outputDir = length + "_" + alpha + "_" + beta + "_" + gamma;

    // Allocate dominance matrix to GPU
    float* dominance = new float[64];
    generateDominance(dominance, params.alpha, params.beta, params.gamma);
    exportDominanceToCSV(dominance, params.species, params);

    float* d_dominance;
    cudaMalloc(&d_dominance, params.species * params.species * sizeof(float));
    cudaMemcpy(d_dominance, dominance, params.species * params.species * sizeof(float), cudaMemcpyHostToDevice);

    if (params.save) {
        std::string mkdirCommand = "mkdir -p " + params.outputDir;
        int result = std::system(mkdirCommand.c_str());
        if (result == 0) {
            std::cout << "Directory created (or already exists): " << params.outputDir << std::endl;
        } else {
            std::cout << "Failed to create directory: " << params.outputDir << std::endl;
        }

        exportParamsToCSV(params);
        exportDominanceToCSV(dominance, params.species, params);
    }

    params.numRandoms = (params.numRandoms / N) * N;

    // std::cout << "------------------- Parameters -------------------\n";
    // std::cout << "MCS: " << params.MCS << "\n";
    // std::cout << "Lattice Length: " << params.L << "\n";
    // std::cout << "Alpha: " << params.alpha << "\n";
    // std::cout << "Beta: " << params.beta << "\n";
    // std::cout << "Gamma: " << params.gamma << "\n";
    // std::cout << "-------------------------------------------------\n";

    // ------------------- CUDA Streams ------------------

    cudaStream_t stream_numbers, stream_steps;
    cudaStreamCreate(&stream_numbers);
    cudaStreamCreate(&stream_steps);

    // ------------------- Random Numbers ------------------

    int index = 0;

    // Device pointers
    float* d_invasion_probabilities;
    int* d_cells;
    int* d_neighbours;

    // Host pointers
    float* h_invasion_probabilities = new float[params.numRandoms];
    int* h_cells = new int[params.numRandoms];
    int* h_neighbours = new int[params.numRandoms];

    cudaError_t err;
    err = cudaMalloc(&d_invasion_probabilities, params.numRandoms * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for invasion probabilities: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMalloc(&d_cells, params.numRandoms * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for cells: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMalloc(&d_neighbours, params.numRandoms * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for neighbours: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Generate random numbers on GPU
    generateRandomNumbers(d_invasion_probabilities, d_cells, d_neighbours, N, params.numRandoms, moore, stream_numbers);
    cudaStreamSynchronize(stream_numbers);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Failed (refreshRandomNumbers): " << cudaGetErrorString(err) << std::endl;
    }

    // Copy memory device -> host
    cudaMemcpy(h_invasion_probabilities, d_invasion_probabilities, params.numRandoms * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells, d_cells, params.numRandoms * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighbours, d_neighbours, params.numRandoms * sizeof(int), cudaMemcpyDeviceToHost);

    // ------------------- Lattice -------------------

    int* d_grid;
    cudaMalloc(&d_grid, N * sizeof(int)); // Allocate memory on GPU

    initialiseGrid(h_grid, params);

    if (params.save) {
        exportGridToCSV(h_grid, params, 0);
    }

    // ------------------- Simulation Parameters -------------------

    GridContext gridCtx;
    StepContext stepCtx;

    stepCtx.cells = new int[N];
    stepCtx.neighbour_dirs = new int[N];
    stepCtx.action_probabilities = new float[N];

    int* d_step_cells;
    int* d_step_neighbour_dirs;
    float* d_step_action_probabilities;

    // Allocate in GPU memory to be used in kernel
    if (params.maxStep) {
    } else {
        cudaMalloc(&d_step_cells, N * sizeof(int));
        cudaMalloc(&d_step_neighbour_dirs, N * sizeof(int));
        cudaMalloc(&d_step_action_probabilities, N * sizeof(float));
    }

    float mu = 1;          // RPSLS interaction
    float sigma = 1;       // Reproduction
    float epsilon = 2 * N; // Migration

    // Normalise the action probabilities
    float sum = mu + sigma + epsilon;
    mu /= sum;
    sigma /= sum;
    epsilon /= sum;

    initialiseGrid(h_grid, params); // Initialise the grid

    std::set<int> speciesSet;
    for (int i = 1; i <= params.species; i++) {
        speciesSet.insert(i);
    }

    // ------------------- Simulation -------------------

    int* d_speciesCounts;
    cudaMalloc(&d_speciesCounts, 8 * sizeof(int));

    int* prev = new int[N];
    std::fill(prev, prev + N, -1);

    if (params.maxStep) {
        int step = params.numRandoms / N;
        for (int mcs = 0; mcs <= MCS; mcs += step) {
            if (compareGrid(h_grid, prev, L, mcs, params)) {
                std::cout << "Stasis reached at MCS: " << mcs << std::endl;
                break;
            }

            if(mcs == 0 || mcs == 1000 || mcs == 2000 || mcs == 3000 || mcs == 5000 ||mcs == 7000 || mcs == 10000 || mcs == 25000 || mcs == 50000 || mcs == 100000 || mcs == 160000 || mcs == 250000 || mcs == 500000){
                writeResults(h_grid, L, mcs, params.alpha, params.beta, params.gamma, false);
                if(params.save){
                    exportGridToCSV(h_grid, params, mcs); 
                }
            }

            if (mcs == MCS) {
                break;
            }

            generateRandomNumbers(d_invasion_probabilities, d_cells, d_neighbours, N, params.numRandoms, moore, stream_numbers);
            cudaMemcpy(prev, h_grid, N * sizeof(int), cudaMemcpyHostToHost);
            maxStep(h_grid, d_grid, d_dominance, d_cells, d_neighbours, d_invasion_probabilities, params, stream_steps);
            cudaStreamSynchronize(stream_numbers);
        }
    } else {                                   // 1MCS per kernel call
        for (int mcs = 0; mcs <= MCS; mcs++) { // Monte Carlos
            // densities(h_grid, d_grid, d_speciesCounts, N, mcs, params.printFrequency, gridCtx, params.species, speciesSet);

            if (compareGrid(h_grid, prev, L, mcs, params)) {
                std::cout << "Stasis reached at MCS: " << mcs << std::endl;
                break;
            }

            if(mcs == 1000 || mcs == 3000 || mcs == 5000 || mcs == 10000){
                writeResults(h_grid, L, mcs, params.alpha, params.beta, params.gamma, -1);
                if(params.save){
                    exportGridToCSV(h_grid, params, mcs); 
                }
            } else if ((mcs <= 100000) && (mcs % 25000 == 0)){
                writeResults(h_grid, L, mcs, params.alpha, params.beta, params.gamma, -1);
                if(params.save){
                    exportGridToCSV(h_grid, params, mcs); 
                }
            }

            if (mcs == MCS) {
                break;
            }

            for (int i = 0; i < N; i++) {
                if (index == 0) {
                    generateRandomNumbers(d_invasion_probabilities, d_cells, d_neighbours, N, params.numRandoms, moore, stream_numbers);
                } else if (index >= params.numRandoms) {
                    cudaStreamSynchronize(stream_numbers);
                    err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        std::cerr << "CUDA Synchronization Failed (refreshRandomNumbers): " << cudaGetErrorString(err) << std::endl;
                    }

                    // Copy memory device -> host
                    cudaMemcpy(h_invasion_probabilities, d_invasion_probabilities, params.numRandoms * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_cells, d_cells, params.numRandoms * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_neighbours, d_neighbours, params.numRandoms * sizeof(int), cudaMemcpyDeviceToHost);
                    index = 0;
                }

                stepCtx.cells[i] = h_cells[index];
                stepCtx.neighbour_dirs[i] = h_neighbours[index];
                stepCtx.action_probabilities[i] = h_invasion_probabilities[index];
                index++;
            }


            cudaMemcpy(d_step_cells, stepCtx.cells, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_step_neighbour_dirs, stepCtx.neighbour_dirs, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_step_action_probabilities, stepCtx.action_probabilities, N * sizeof(float), cudaMemcpyHostToDevice);
            step(h_grid, d_grid, d_dominance, d_step_cells, d_step_neighbour_dirs, d_step_action_probabilities, params, stream_steps);
        }
    }

    // ------------------- Write Results -------------------
    writeAllResultsToFile(params.resultsFile);
    // ------------------- Free Memory -------------------

    // Grids
    delete[] h_grid;
    cudaFree(d_grid);

    // Random numbers
    delete[] h_invasion_probabilities;
    delete[] h_cells;
    delete[] h_neighbours;
    cudaFree(d_invasion_probabilities);
    cudaFree(d_cells);
    cudaFree(d_neighbours);

    if (!params.maxStep) {
        delete[] stepCtx.cells;
        delete[] stepCtx.neighbour_dirs;
        delete[] stepCtx.action_probabilities;
    }

    cudaFree(d_step_cells);
    cudaFree(d_step_neighbour_dirs);
    cudaFree(d_step_action_probabilities);

    // Densities
    cudaFree(d_speciesCounts);

    cudaStreamDestroy(stream_numbers);
    cudaStreamDestroy(stream_steps);

    return 0;
}