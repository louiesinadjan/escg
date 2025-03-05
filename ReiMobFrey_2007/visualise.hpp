#pragma once
#include "external/matplotlib-cpp/matplotlibcpp.h"
#include <vector>
#include <cmath>
#include <filesystem> 


namespace plt = matplotlibcpp;

// Function to plot densities over Monte Carlo steps
void plot_densities(const std::vector<double>& steps,
                    const std::vector<double>& densityRock,
                    const std::vector<double>& densityPaper,
                    const std::vector<double>& densityScissors,
                    const std::vector<double>& densityLizard,
                    const std::vector<double>& densitySpock);

// Function to plot a snapshot of the grid at a given Monte Carlo step
void plot_snapshot(const int* grid, int L, int H, bool moore, int mcs, float mobility);

