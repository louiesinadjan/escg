#include "visualise.hpp"
#include <sstream>

void plot_densities(GridContext g, Params p) {

    plt::figure_size(2000, 800);
    plt::tight_layout();

    // Plot each density with a different color
    plt::semilogx(g.steps, g.densityRock, "b-");
    plt::semilogx(g.steps, g.densityPaper, "c-");
    plt::semilogx(g.steps, g.densityScissors, "g-");
    plt::semilogx(g.steps, g.densityLizard, "y-");
    plt::semilogx(g.steps, g.densitySpock, "r-");

    // Add legend (labels in the same order as plots)
    plt::legend();

    // Axis labels and title
    plt::xlabel("Steps");
    plt::ylabel("$\\rho_i$"); // LaTeX style for rho_i
    plt::title("Density Evolution Over Time");

    bool moore = p.neighbourhood == 8;
    std::string length = "l" + std::to_string(p.L);
    std::string height = "h" + std::to_string(p.H);
    std::string neighbourhood = moore ? "Moore" : "VN";

    // Convert to scientific notation
    std::ostringstream oss;
    oss.precision(2);
    oss << std::scientific << p.mobility;
    std::string mobility_str = "M" + oss.str();

    std::string species = std::to_string(p.species) + " species";

    plt::save("densities_" + length + "_" + height + "_" + neighbourhood + "_" + mobility_str + "_" + species + ".png");
}

// https://github.com/lava/matplotlib-cpp/blob/master/examples/imshow.cpp
void plot_snapshot(const int* grid, int mcs, Params p) {
    bool moore = p.neighbourhood == 8;

    plt::figure_size(2000, 2000);
    plt::tight_layout();

    plt::axis("off"); // disables the axis labels and ticks

    // Flatten the grid into a 1D vector of floats
    std::vector<float> flatGrid(p.L * p.H);
    for (int i = 0; i < p.H * p.L; i++) {
        flatGrid[i] = static_cast<float>(grid[i]);
    }

    // Pointer to the flattened grid data
    const float* gridPtr = &(flatGrid[0]);

    // Plot using imshow
    plt::figure();
    plt::imshow(gridPtr, p.H, p.L, 1);

    // plt::title("Snapshot at MCS = " + std::to_string(mcs));

    // plt::show();
    std::string length = "l" + std::to_string(p.L);
    std::string height = "h" + std::to_string(p.H);
    std::string neighbourhood = moore ? "Moore" : "VN";
    std::string mcs_str = "mcs" + std::to_string(mcs);

    // Convert to scientific notation
    std::ostringstream oss;
    oss.precision(2);
    oss << std::scientific << p.mobility;
    std::string mobility_str = "M" + oss.str();

    std::string species = std::to_string(p.species) + " species";

    plt::title(length + "_" + height + "_" + neighbourhood + "_" + mobility_str + "_" + mcs_str);
    plt::save("snapshot_" + length + "_" + height + "_" + neighbourhood + "_" + mobility_str + "_" + mcs_str + "_" + species + ".png");
}