#pragma once

#include "config.hpp"
#include "simulation.hpp" // Include for Species and grid data

namespace plt = matplotlibcpp;


// Function to plot densities over Monte Carlo steps
void plot_densities(GridContext g, Params p);

// Function to plot a snapshot of the grid at a given Monte Carlo step
void plot_snapshot(const int* grid, int mcs, Params p);