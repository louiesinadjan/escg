#include "config.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void exportGridToCSV(int* grid, Params p, int mcs);
void exportParamsToCSV(Params p);

int importCSVToGrid(int* grid, int N); // Returns the MCS to resume from

void exportDominanceToCSV(float* dominance, int species, Params p);

void generateDominance(float* dominance, float alpha, float beta, float gamma);

void writeResults(int* grid, int L, int mcs, float alpha, float beta, float gamma, int stable);