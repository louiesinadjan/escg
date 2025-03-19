#pragma once
#include "external/matplotlib-cpp/matplotlibcpp.h"
#include <cmath>
#include <filesystem>
#include <vector>
#include "config.hpp"

namespace plt = matplotlibcpp;

// Function to plot densities over Monte Carlo steps
void plot_densities(GridContext g, Params p);

// Function to plot a snapshot of the grid at a given Monte Carlo step
void plot_snapshot(const int* grid, int mcs, Params p);

