//
//  main.cpp
//  single-threaded-escg
//
//  Created by Louie Sinadjan on 11/01/2025.
//

#include "simulation.hpp"
#include "visualise.hpp"

int main(int argc, const char* argv[]) {
    int MCS = 6000; // 100,000  Monte Carlo Steps

    int L = 200;        // Length of lattice
    int N = L * L;      // Elementary time steps
    float M = 1e-6f;    // Mobility 'since it is proportional to the typical area
                        // explored by one mobile individual per unit time'
    int grid[200][200]; // Grid size = 200 x 200

    float mu = 1;              // RPSLS interaction
    float sigma = 1;           // Reproduction
    float epsilon = 2 * M * N; // Migration



    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // std::uniform_int_distribution<int> dist(0, 5); // Range: 0 to 5 (EMPTY + RPSLS)
    std::uniform_int_distribution<int> dist(1, 5); // Range: 1 to 5 (RPSLS)

    // Randomly initialise the grid with species (stored as int)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i][j] = dist(gen); // Randomly assign a species (or EMPTY) as an integer
        }
    }

    for (int mcs = 0; mcs <= MCS; mcs++) { // Monte Carlo Steps
        densities(grid, L, mcs);           // Every MCS, call densities to add to density vectors for visualisation after simulation

        if (mcs == 0 || mcs == 2000 || mcs == 6000 || mcs == 20000 || mcs == 100000) {
            plot_snapshot(grid, L, mcs);
        }

        for (int n = 0; n < N; n++) { // Elementary Time Steps
            step(L, grid, mu, sigma, epsilon);
        }
    }

    show(); // Plot density against steps

    std::cout << "Simulation Complete." << std::endl;

    return 0;
}
