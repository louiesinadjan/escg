#pragma once
#include "external/matplotlib-cpp/matplotlibcpp.h"
#include <vector>
#include "simulation.hpp" // Include for Species and grid data

namespace plt = matplotlibcpp;

// Function to plot densities over Monte Carlo steps
void plot_densities(const std::vector<int>& steps,
                    const std::vector<double>& densityRock,
                    const std::vector<double>& densityPaper,
                    const std::vector<double>& densityScissors,
                    const std::vector<double>& densityLizard,
                    const std::vector<double>& densitySpock);

// Function to plot a snapshot of the grid at a given Monte Carlo step
void plot_snapshot(const Species grid[200][200], int L, int mcs);

#endif // VISUALISE_HPP