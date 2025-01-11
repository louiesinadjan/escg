#include "simulation.hpp"

bool dominates(Species specie, Species neighbour) {
    // RPSLS dominance rules
    switch (specie) {
        case Species::ROCK: // Crushes SCISSORS, LIZARD
            return (neighbour == Species::SCISSORS || neighbour == Species::LIZARD);
        case Species::PAPER: // Covers ROCK, disproves SPOCK
            return (neighbour == Species::ROCK || neighbour == Species::SPOCK);
        case Species::SCISSORS: // Cuts PAPER, decapitates LIZARD
            return (neighbour == Species::PAPER || neighbour == Species::LIZARD);
        case Species::LIZARD: // Poisons SPOCK, eats PAPER
            return (neighbour == Species::SPOCK || neighbour == Species::PAPER);
        case Species::SPOCK: // Smashes SCISSORS, vaporises ROCK
            return (neighbour == Species::SCISSORS || neighbour == Species::ROCK);
        default:
            return false;
    }
}

// Helper function to wrap around the grid when selecting a neighbour
int wrap(int index, int L) { return (index + L) % L; }

void step(int L, Species grid[200][200], float mu, float sigma, float epsilon) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_pos(0, L - 1);     // Random position in grid
    std::uniform_int_distribution<int> dist_dir(0, 7);         // Random neighbour direction
    std::uniform_real_distribution<float> dist_prob(0.0, 1.0); // Random probability for actions

    int dX[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dY[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    int i = dist_pos(gen);
    int j = dist_pos(gen);
    Species specie = grid[i][j];

    int dir = dist_dir(gen);
    int ni = wrap(i + dX[dir], L);
    int nj = wrap(j + dY[dir], L);
    Species neighbour = grid[ni][nj];

    float random_action = dist_prob(gen);

    if (random_action < mu && specie != neighbour) {
        if (dominates(neighbour, specie)) {
            grid[i][j] = neighbour; // Neighbour dominates and replaces specie
        } else if (dominates(specie, neighbour)) {
            grid[ni][nj] = specie; // Specie dominates and replaces neighbour
        }
    } else if (random_action < mu + sigma && neighbour == Species::EMPTY) {
        grid[ni][nj] = specie; // Reproduction
    } else if (random_action < mu + sigma + epsilon) {
        std::swap(grid[i][j], grid[ni][nj]); // Migration
    }
}

void densities(Species grid[200][200], int L, int mcs) {
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
                case Species::EMPTY:
                    emptyCounter++;
                    break;
                case Species::ROCK:
                    rockCounter++;
                    break;
                case Species::PAPER:
                    paperCounter++;
                    break;
                case Species::SCISSORS:
                    scissorsCounter++;
                    break;
                case Species::LIZARD:
                    lizardCounter++;
                    break;
                case Species::SPOCK:
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