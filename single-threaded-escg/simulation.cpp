#include "simulation.hpp"
#include "visualise.hpp"
#include <algorithm>

bool dominates(int specie, int neighbour, int speciesNum, int* dominance) {
    if (specie == 0 || neighbour == 0) {
        return false; // Empty spaces don't dominate anything
    }
    return dominance[((specie - 1) * speciesNum + (neighbour - 1))] == 1;
}

// Helper function to wrap around the grid when selecting a neighbour
int wrap(int index, int L) { return (index + L) % L; }

void step(Params p, int* grid, int* dominance, float mu, float sigma, float epsilon) {
    static std::random_device rd;
    static std::mt19937 gen(rd());                                              // Random number generator
    std::uniform_int_distribution<int> dist_pos(0, (p.L * p.L) - 1);            // Random position in grid
    std::uniform_int_distribution<int> dist_dirVN(0, 3);                        // Random neighbour direction (0 to 3 for four neighbours)
    std::uniform_int_distribution<int> dist_dirM(0, 7);                         // Random neighbour direction (0 to 3 for four neighbours)
    std::uniform_real_distribution<float> dist_prob(0.0, mu + sigma + epsilon); // Random probability for actions

    int offsets[8][2] = {
        // Neighbour offsets for 8-way neighbourhood
        {-1, 0},  // Up
        {1, 0},   // Down
        {0, -1},  // Left
        {0, 1},   // Right
        {-1, -1}, // Up-Left
        {-1, 1},  // Up-Right
        {1, -1},  // Down-Left
        {1, 1}    // Down-Right
    };

    int i = dist_pos(gen); // Random position in grid
    int specie = grid[i];  // Specie at position (i, j)

    int dir;
    if (p.neighbourhood == 4) {
        dir = dist_dirVN(gen); // Random neighbour direction
    } else {
        dir = dist_dirM(gen); // Random neighbour direction
    }

    // Convert the 1D cell index to 2D coordinates.
    int row = i / p.L;
    int col = i % p.L;

    // Compute the new row and column based on the direction.
    int n_row, n_col;
    if (p.flux) {
        n_row = (row + offsets[dir][0] + p.L) % p.L;
        n_col = (col + offsets[dir][1] + p.L) % p.L;
    } else {
        n_row = row + offsets[dir][0];
        if (n_row < 0 || n_row >= p.H) {
            n_row = row - offsets[dir][0];
        }
        n_col = col + offsets[dir][1];
        if (n_col < 0 || n_col >= p.L) {
            n_col = col - offsets[dir][1];
        }
    }

    // Convert the new row and column back into a 1D index.
    int n_i = n_row * p.L + n_col;
    int neighbour = grid[n_i];            // Specie at neighbour position (ni, nj)
    float random_action = dist_prob(gen); // Random probability for actions

    if (specie == neighbour) {
        return; // Do nothing if the specie and neighbour are the same
    }

    // RPSLS Interaction, Reproduction, Migration actions with probabilities mu, sigma, epsilon
    if (random_action < epsilon) { // Migration
        std::swap(grid[i], grid[n_i]);
    } else if (random_action < epsilon + mu) {                        // RPSLS Interaction selected
        if (specie != neighbour && neighbour != 0) {                  // Empty neighbours do not partake in RPSLS interaction
            if (dominates(specie, neighbour, p.species, dominance)) { // Neighbour dominates, specie becomes empty
                grid[n_i] = 0;
            } else if (dominates(neighbour, specie, p.species, dominance)) { // Specie dominates, neighbour becomes empty
                grid[i] = 0;
            }
        }
    } else if (random_action < epsilon + mu + sigma) { // Reproduction selected
        if (neighbour == 0) {                          // Only possible if the neighbour is empty and the specie is not
            grid[n_i] = specie;
        } else if (specie == 0) {
            grid[i] = neighbour;
        }
    } else {
        // Do nothing - step completed
    }
}

void densities(GridContext& gridCtx, int* grid, Params p, int mcs) {
    int N = p.L * p.L;
    int* speciesCounts = new int[p.species + 1](); // EMPTY + Species

    // Iterate through the entire grid and count each species
    for (int i = 0; i < p.L * p.L; i++) {
        speciesCounts[grid[i]]++;
    }

    double emptyDensity = (static_cast<double>(speciesCounts[0]) / N) * 100;

    std::vector<double> densities;
    for (int i = 1; i <= p.species; i++) {
        densities.push_back((static_cast<double>(speciesCounts[i]) / N) * 100);
    }

    gridCtx.steps.push_back(mcs);
    gridCtx.speciesDensities.push_back(densities);

    // Print the densities
    if (mcs % p.printFrequency == 0) {
        std::cout << "Population Densities at: " << mcs << std::endl;
        std::cout << "EMPTY: " << emptyDensity << std::endl;

        for (int i = 0; i < p.species; i++) {
            // The vector is 0-indexed but the species starts from Species 1 (value 0 in grid refers to empty)
            std::cout << "Species " << (i + 1) << ": " << densities[i] << std::endl;
        }
        std::cout << std::endl;
    }

    delete[] speciesCounts;
}
