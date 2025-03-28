#include "config.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void generateDominance(float* dominance, float alpha, float beta, float gamma);
void writeResults(int* grid, int L, int mcs, float alpha, float beta, bool stable);
int importCSVToGrid(int* grid, int N);