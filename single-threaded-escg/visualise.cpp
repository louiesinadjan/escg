#include "visualise.hpp"

void plot_densities(const std::vector<int>& steps,
                    const std::vector<double>& densityRock,
                    const std::vector<double>& densityPaper,
                    const std::vector<double>& densityScissors,
                    const std::vector<double>& densityLizard,
                    const std::vector<double>& densitySpock) {
    plt::figure();
    plt::xscale("log"); // Logarithmic scale for Monte Carlo steps
    plt::plot(steps, densityRock, {{"label", "Rock"}, {"color", "red"}});
    plt::plot(steps, densityPaper, {{"label", "Paper"}, {"color", "green"}});
    plt::plot(steps, densityScissors, {{"label", "Scissors"}, {"color", "blue"}});
    plt::plot(steps, densityLizard, {{"label", "Lizard"}, {"color", "yellow"}});
    plt::plot(steps, densitySpock, {{"label", "Spock"}, {"color", "cyan"}});

    plt::xlabel("Monte Carlo Steps");
    plt::ylabel("Density");
    plt::title("Species Densities Over Time");
    plt::legend();
    plt::grid(true);
    plt::show();
}

void plot_snapshot(const Species grid[200][200], int L, int mcs) {
    std::vector<std::vector<int>> gridData(L, std::vector<int>(L));

    // Convert the grid to integer representation for plotting
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            gridData[i][j] = static_cast<int>(grid[i][j]);
        }
    }

    // Plot the grid snapshot
    plt::figure();
    plt::imshow(gridData, {{"cmap", "jet"}});
    plt::title("Snapshot at MCS = " + std::to_string(mcs));
    plt::colorbar();
    plt::show();
}