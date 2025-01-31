#include "simulation.hpp"
#include "visualise.hpp"

std::vector<double> steps;
std::vector<double> densityRock;
std::vector<double> densityPaper;
std::vector<double> densityScissors;
std::vector<double> densityLizard;
std::vector<double> densitySpock;

float migrationCounter = 0;
float interactionCounter = 0;
float reproductionCounter = 0;

bool dominates(int specie, int neighbour) {
    switch (specie) {
        case 1: // ROCK (crushes 3: SCISSORS, crushes 4: LIZARD)
            // return (neighbour == 3 || neighbour == 4);
            return neighbour == 4; // Absence of Rock - Scissors interaction

        case 2: // PAPER (covers 1: ROCK, disproves 5: SPOCK)
            return (neighbour == 1 || neighbour == 5);

        case 3: // SCISSORS (cuts 2: PAPER, decapitates 4: LIZARD)
            return (neighbour == 2 || neighbour == 4);

        case 4: // LIZARD (poisons 5: SPOCK, eats 2: PAPER)
            return (neighbour == 5 || neighbour == 2);

        case 5: // SPOCK (smashes 3: SCISSORS, vaporises 1: ROCK)
            return (neighbour == 3 || neighbour == 1);

        default: // 0 (EMPTY) or any invalid integer
            return false;
    }
}

// Helper function to wrap around the grid when selecting a neighbour
int wrap(int index, int L) { return (index + L) % L; }

void step(int L, int grid[200][200], int cell, int neighbourDir, int action) {
    int dX[4] = {-1, 1, 0, 0}; // Neighbour directions (up, down, left, right)
    int dY[4] = {0, 0, -1, 1}; // Neighbour directions (up, down, left, right)

    int i = cell / L; // Row in grid
    int j = cell % L; // Column in grid

    int specie = grid[i][j]; // Specie at position (i, j)

    // print grid coordinate then specie
    //  std::cout << "i:\t" << i << " j:\t" << j << "\t\tspecie: " << specie << std::endl;

    int ni = wrap(i + dX[neighbourDir], L); // Neighbour position in grid
    int nj = wrap(j + dY[neighbourDir], L); // Neighbour position in grid
    int neighbour = grid[ni][nj];           // Neighbour specie

    if (action == 1) { // RPSLS Interaction Selected
        interactionCounter++;
        if (dominates(specie, neighbour)) {
            grid[ni][nj] = 0; // Neighbour  becomes empty
        } else if (dominates(neighbour, specie)) {
            grid[i][j] = 0; // Specie becomes empty
        }
    } else if (action == 2) { // Reproduction Selected
        reproductionCounter++;
        if (neighbour == 0 && specie != 0) {           // Specie reproduces into empty neighbour
            grid[ni][nj] = specie;
            reproduction++;
        } else if (specie == 0 && neighbour != 0) { // Neighbour reproduces into empty specie
            grid[i][j] = neighbour;
            reproduction++;
        }
    } else if (action == 3) { // Migration Selected
        migrationCounter++;
        std::swap(grid[i][j], grid[ni][nj]);
    } else {
        std::cout << "Invalid action selected" << std::endl;
    }
}

void densities(int grid[200][200], int L, int mcs) {
    int emptyCounter = 0;
    int rockCounter = 0;
    int paperCounter = 0;
    int scissorsCounter = 0;
    int lizardCounter = 0;
    int spockCounter = 0;

    // Iterate through the entire grid and count each species
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            switch (grid[i][j]) {
                case 0:
                    emptyCounter++;
                    break;
                case 1:
                    rockCounter++;
                    break;
                case 2:
                    paperCounter++;
                    break;
                case 3:
                    scissorsCounter++;
                    break;
                case 4:
                    lizardCounter++;
                    break;
                case 5:
                    spockCounter++;
                    break;
            }
        }
    }

    // Calculate densities as percentages of the total grid size
    int totalCells = L * L;
    double emptyDensity = (static_cast<double>(emptyCounter) / totalCells) * 100;
    double rockDensity = (static_cast<double>(rockCounter) / totalCells) * 100;
    double paperDensity = (static_cast<double>(paperCounter) / totalCells) * 100;
    double scissorsDensity = (static_cast<double>(scissorsCounter) / totalCells) * 100;
    double lizardDensity = (static_cast<double>(lizardCounter) / totalCells) * 100;
    double spockDensity = (static_cast<double>(spockCounter) / totalCells) * 100;

    steps.push_back(mcs);
    densityRock.push_back(rockDensity);
    densityPaper.push_back(paperDensity);
    densityScissors.push_back(scissorsDensity);
    densityLizard.push_back(lizardDensity);
    densitySpock.push_back(spockDensity);

    // Print the densities
    if (mcs % 200 == 0) {
        std::cout << "Population Densities at: " << mcs << std::endl;
        std::cout << "EMPTY: " << emptyDensity << std::endl;
        std::cout << "ROCK: " << rockDensity << std::endl;
        std::cout << "PAPER: " << paperDensity << std::endl;
        std::cout << "SCISSORS: " << scissorsDensity << std::endl;
        std::cout << "LIZARD: " << lizardDensity << std::endl;
        std::cout << "SPOCK: " << spockDensity << std::endl;
    }
}

void show() {
    plot_densities(steps, densityRock, densityPaper, densityScissors, densityLizard, densitySpock);
    std::cout << "Migration: " << migrationCounter << std::endl;
    std::cout << "Interaction: " << interactionCounter << std::endl;
    std::cout << "Reproduction: " << reproductionCounter << std::endl;

    std::cout << "Migration: " << migrationCounter / (migrationCounter + interactionCounter + reproductionCounter) * 100 << std::endl;
    std::cout << "Interaction: " << interactionCounter / (migrationCounter + interactionCounter + reproductionCounter) * 100 << std::endl;
    std::cout << "Reproduction: " << reproductionCounter / (migrationCounter + interactionCounter + reproductionCounter) * 100 << std::endl;
}