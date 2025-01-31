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
bool dominates(int specie, int neighbour);                                      // RPSLS interaction
int wrap(int index, int L);                                                     // Helper function to make the grid like a torus
void step(int L, int grid[200][200], int specie, int neighbourDir, int action); // ELementary time step
void densities(int grid[200][200], int L, int mcs);                             // RPSLS densities
void show();                                                                    // Show visualisations