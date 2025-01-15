#include "visualise.hpp"

void plot_densities(const std::vector<int>& steps, const std::vector<double>& densityRock, const std::vector<double>& densityPaper, const std::vector<double>& densityScissors,
                    const std::vector<double>& densityLizard, const std::vector<double>& densitySpock) {}

void plot_snapshot(const int grid[200][200], int L, int mcs) {
    // Flatten the grid into a 1D vector of floats
    std::vector<float> flatGrid(L * L);
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            flatGrid[i * L + j] = static_cast<float>(grid[i][j]);
        }
    }

    // Pointer to the flattened grid data
    const float* gridPtr = &(flatGrid[0]);

    // Plot using imshow
    plt::figure();
    plt::imshow(gridPtr, L, L, 1);
    plt::title("Snapshot at MCS = " + std::to_string(mcs));
    plt::show();
}