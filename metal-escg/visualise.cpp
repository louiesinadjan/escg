#include "visualise.hpp"
#include <sstream>

void plot_densities(GridContext g, Params p) {
    plt::figure_size(2000, 800);
    plt::tight_layout();

    // Define a set of colors to cycle through
    std::vector<std::string> colors = {"blue", "cyan", "green", "yellow", "r-", "m-", "k-", "orange", "pink", "purple"};

    // Ensure speciesDensities is not empty
    if (!g.speciesDensities.empty()) {
        size_t speciesCount = g.speciesDensities[0].size(); // Get number of species

        for (size_t i = 0; i < speciesCount; i++) {
            std::string color = colors[i % colors.size()]; // Cycle through colors
            std::vector<double> speciesDensity;

            // Extract the density evolution for species `i`
            for (size_t step = 0; step < g.speciesDensities.size(); step++) {
                speciesDensity.push_back(g.speciesDensities[step][i]);
            }

            plt::semilogx(g.steps, speciesDensity, color);
        }

        plt::legend();
    }
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
    std::string flux = p.flux ? "flux" : "noflux";
    std::string species = std::to_string(p.species) + "species";

    plt::title(length + "_" + height + "_" + neighbourhood + "_" + mobility_str + "_" + flux + "_" + species);
    plt::save("./" + p.outputDir + "/densities.png");
    plt::close();
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

    // https: // matplotlib.org/stable/users/explain/colors/colormaps.html
    // Change colormap as desired
    std::map<std::string, std::string> keywords;

    if(p.species == 3){
        keywords = {{"cmap", "Accent"}};
    } else if (p.species <= 5) {
        keywords = {{"cmap", "cividis"}};
    } else {
        keywords = {{"cmap", "tab20b"}};
    }

    plt::imshow(gridPtr, p.H, p.L, 1, keywords);
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
    std::string flux = p.flux ? "flux" : "noflux";
    std::string species = std::to_string(p.species) + "species";

    plt::title(length + "_" + height + "_" + neighbourhood + "_" + mobility_str + "_" + flux + "_" + species + "_" + mcs_str);

    plt::save("./" + p.outputDir + "/ss_" + mcs_str + ".png");
    plt::close();
}