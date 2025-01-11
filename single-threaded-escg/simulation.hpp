//
//  Simulation.h
//  single-threaded-escg
//
//  Created by Louie Sinadjan on 11/01/2025.
//

#pragma once
#include <algorithm> // For std::swap
#include <iostream>
#include <random>

// Enum for species in the RPSLS simulation
enum class Species { EMPTY, ROCK, PAPER, SCISSORS, LIZARD, SPOCK };

// Function declarations
bool dominates(Species specie, Species neighbour);
int wrap(int index, int L);
void step(int L, Species grid[200][200], float mu, float sigma, float epsilon);
void densities(Species grid[200][200], int L, int mcs);