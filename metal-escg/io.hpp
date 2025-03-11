#include "config.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void exportGridToCSV(int* grid, Params p, int mcs);
void exportParamsToCSV(Params p);

int importCSVToGrid(int* grid, int N); // Returns the MCS to resume from
void importCSVToParams(Params& p); // Imports the parameters from the csv file

void importCSVToDominance(int* dominates, int species);
void exportDominanceToCSV(int* dominates, int species, Params p);