#include "simulation.hpp"

bool dominates(int specie, int neighbour) {
    switch (specie) {
        case 1: // ROCK (crushes 3: SCISSORS, 4: LIZARD)
            return (neighbour == 3 || neighbour == 4);
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

void step(int L, int grid[200][200], float mu, float sigma, float epsilon) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_pos(0, L - 1);     // Random position in grid
    std::uniform_int_distribution<int> dist_dir(0, 7);         // Random neighbour direction
    std::uniform_real_distribution<float> dist_prob(0.0, 1.0); // Random probability for actions

    int dX[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dY[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    int i = dist_pos(gen);
    int j = dist_pos(gen);
    int specie = grid[i][j];

    int dir = dist_dir(gen);
    int ni = wrap(i + dX[dir], L);
    int nj = wrap(j + dY[dir], L);
    int neighbour = grid[ni][nj];

    float random_action = dist_prob(gen);

    if (random_action < mu && specie != neighbour) {
        if (dominates(neighbour, specie)) {
            grid[i][j] = neighbour; // Neighbour dominates and replaces specie
        } else if (dominates(specie, neighbour)) {
            grid[ni][nj] = specie; // Specie dominates and replaces neighbour
        }
    } else if (random_action < mu + sigma && neighbour == static_cast<int>(Species::EMPTY)) {
        grid[ni][nj] = specie; // Reproduction
    } else if (random_action < mu + sigma + epsilon) {
        std::swap(grid[i][j], grid[ni][nj]); // Migration
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
    float emptyDensity = (static_cast<float>(emptyCounter) / totalCells) * 100;
    float rockDensity = (static_cast<float>(rockCounter) / totalCells) * 100;
    float paperDensity = (static_cast<float>(paperCounter) / totalCells) * 100;
    float scissorsDensity = (static_cast<float>(scissorsCounter) / totalCells) * 100;
    float lizardDensity = (static_cast<float>(lizardCounter) / totalCells) * 100;
    float spockDensity = (static_cast<float>(spockCounter) / totalCells) * 100;

    // Print the densities
    std::cout << "Population Densities at: " << mcs << std::endl;
    std::cout << "EMPTY: " << emptyDensity << std::endl;
    std::cout << "ROCK: " << rockDensity << std::endl;
    std::cout << "PAPER: " << paperDensity << std::endl;
    std::cout << "SCISSORS: " << scissorsDensity << std::endl;
    std::cout << "LIZARD: " << lizardDensity << std::endl;
    std::cout << "SPOCK: " << spockDensity << std::endl;
}