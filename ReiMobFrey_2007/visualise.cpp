#include "visualise.hpp"

void plot_densities(const std::vector<double>& steps, const std::vector<double>& densityRock, const std::vector<double>& densityPaper, const std::vector<double>& densityScissors,
                    const std::vector<double>& densityLizard, const std::vector<double>& densitySpock) {

    plt::figure_size(2000, 800);
    plt::tight_layout();

    // Plot each density with a different color
    plt::semilogx(steps, densityRock, "r-");
    plt::semilogx(steps, densityPaper, "g-");
    plt::semilogx(steps, densityScissors, "b-");
    plt::semilogx(steps, densityLizard, "y-");
    plt::semilogx(steps, densitySpock, "m-");

    // Add legend (labels in the same order as plots)
    plt::legend();

    // Axis labels and title
    plt::xlabel("Steps");
    plt::ylabel("$\\rho_i$"); // LaTeX style for rho_i
    plt::title("Density Evolution Over Time");

    // Display the plot

    // plt::show();

    plt::save("densities.png");
}

// https://github.com/lava/matplotlib-cpp/blob/master/examples/imshow.cpp
void plot_snapshot(const int* grid, int L, int H, bool moore, int mcs) {
    plt::figure_size(2000, 2000);
    plt::tight_layout();

    // Flatten the grid into a 1D vector of floats
    std::vector<float> flatGrid(L * H);
    for (int i = 0; i < H * L; i++) {
        flatGrid[i] = static_cast<float>(grid[i]);
    }

    // Pointer to the flattened grid data
    const float* gridPtr = &(flatGrid[0]);

    // Plot using imshow
    plt::figure();
    plt::imshow(gridPtr, H, L, 1);

    // plt::title("Snapshot at MCS = " + std::to_string(mcs));

    // plt::show();
    std::string length = "l" + std::to_string(L);
    std::string height = "h" + std::to_string(H);
    std::string neighbourhood = moore ? "moore" : "vonNeumann";
    std::string mcs_str = "mcs" + std::to_string(mcs);

    plt::title("Length = " + std::to_string(L) + ", Height = " + std::to_string(H) + ", Neighbourhood = " + neighbourhood + ", MCS = " + std::to_string(mcs));
    plt::save("snapshot_" + length + "_" + height + "_" + neighbourhood + "_" + mcs_str + ".png");
}