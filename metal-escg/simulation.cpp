#include "simulation.hpp"
#include "visualise.hpp"

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

    std::string direction[] = {"UP", "DOWN", "LEFT", "RIGHT"};

    int i = cell / L; // Row in grid
    int j = cell % L; // Column in grid

    int specie = grid[i][j]; // Specie at position (i, j)

    int ni = wrap(i + dX[neighbourDir], L); // Neighbour position in grid
    int nj = wrap(j + dY[neighbourDir], L); // Neighbour position in grid
    int neighbour = grid[ni][nj];           // Neighbour specie

    if (action == 1) { // RPSLS Interaction Selected
        if (dominates(specie, neighbour)) {
            grid[ni][nj] = 0; // Neighbour  becomes empty
        } else if (dominates(neighbour, specie)) {
            grid[i][j] = 0; // Specie becomes empty
        } else {
        }
    } else if (action == 2) {                // Reproduction Selected
        if (neighbour == 0 && specie != 0) { // Specie reproduces into empty neighbour
            grid[ni][nj] = specie;
        } else if (specie == 0 && neighbour != 0) { // Neighbour reproduces into empty specie
            grid[i][j] = neighbour;
        } else {
        }
    } else if (action == 3) { // Migration Selected
        std::swap(grid[i][j], grid[ni][nj]);
    } else {
        std::cout << "Invalid action selected" << std::endl;
    }
}

