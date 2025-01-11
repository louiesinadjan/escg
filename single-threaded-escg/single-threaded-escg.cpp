//
//  main.cpp
//  single-threaded-escg
//
//  Created by Louie Sinadjan on 11/01/2025.
//

#include "simulation.hpp"

int main(int argc, const char* argv[]) {
    int MCS = 60000;

    int L = 200;            // Length of lattice
    int N = L * L;          // Elementary time steps
    float M = 1e-6f;        // Mobility 'since it is proportional to the typical area
                            // explored by one mobile individual per unit time'
    Species grid[200][200]; // Grid size = 200 x 200

    float mu = 1;              // RPSLS interaction
    float sigma = 1;           // Reproduction
    float epsilon = 2 * M * N; // Migration

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 5); // Range: 0 to 5 (EMPTY + RPSLS)

    // Randomly initialise the grid with species
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i][j] = static_cast<Species>(dist(gen)); // Randomly assign a species or EMPTY
        }
    }

    for (int mcs = 0; mcs <= MCS; mcs++) { // Monte Carlo Steps
        if (mcs % 1000 == 0) {
            densities(grid, L, mcs);
        } else if (mcs % 100 == 0) {
            std::cout << "MCS = " << mcs << std::endl;
        }
        
        for (int n = 0; n < N; n++) { // Elementary Time Steps
            step(L, grid, mu, sigma, epsilon);
        }
    }

    std::cout << "Simulation Complete.";
    return 0;
}
