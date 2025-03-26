//
//  Simulation.h
//  single-threaded-escg
//
//  Created by Louie Sinadjan on 11/01/2025.
//

#pragma once
#include "config.hpp"

// Enum for species in the RPSLS simulation
enum class Species { EMPTY, ROCK, PAPER, SCISSORS, LIZARD, SPOCK };

// Function declarations
bool dominates(int specie, int neighbour, int speciesNum, int* dominance);                         // RPSLS interaction
int wrap(int index, int L);                                        // Helper function to make the grid like a torus
void step(Params p, int* grid, int* dominance, float mu, float sigma, float epsilon); // Elementary time step
void densities(GridContext& gridCtx, int* grid, Params p, int mcs); // RPSLS densities
