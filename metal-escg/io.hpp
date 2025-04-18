#include "config.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void exportGridToCSV(int* grid, Params p, int mcs);
void exportParamsToCSV(Params p);

int importCSVToGrid(int* grid, int N); // Returns the MCS to resume from
void importCSVToParams(Params& p); // Imports the parameters from the csv file

int importCSVToDominance(int* &dominance);
void exportDominanceToCSV(int* dominance, int species, Params p);

void generateCirculantAdjacencyMatrix(int* dominance, int speciesCount);