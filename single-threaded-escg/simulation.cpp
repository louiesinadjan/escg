#include "simulation.hpp"
#include "visualise.hpp"

std::vector<double> steps;
std::vector<double> densityRock;
std::vector<double> densityPaper;
std::vector<double> densityScissors;
std::vector<double> densityLizard;
std::vector<double> densitySpock;

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

void step(int L, int grid[200][200], float mu, float sigma, float epsilon, int& migration, int& reproduction, int& interaction) {
    static std::random_device rd;
    static std::mt19937 gen(rd());                                              // Random number generator
    std::uniform_int_distribution<int> dist_pos(0, L - 1);                      // Random position in grid
    std::uniform_int_distribution<int> dist_dir(0, 3);                          // Random neighbour direction (0 to 3 for four neighbours)
    std::uniform_real_distribution<float> dist_prob(0.0, mu + sigma + epsilon); // Random probability for actions

    int dX[4] = {-1, 1, 0, 0}; // Neighbour directions (up, down, left, right)
    int dY[4] = {0, 0, -1, 1}; // Neighbour directions (up, down, left, right)

    int i = dist_pos(gen);   // Random position in grid
    int j = dist_pos(gen);   // Random position in grid
    int specie = grid[i][j]; // Specie at position (i, j)

    int dir = dist_dir(gen);       // Random neighbour direction
    int ni = wrap(i + dX[dir], L); // Neighbour position in grid
    int nj = wrap(j + dY[dir], L); // Neighbour position in grid
    int neighbour = grid[ni][nj];  // Neighbour specie

    float random_action = dist_prob(gen); // Random probability for actions

    // RPSLS Interaction, Reproduction, Migration actions with probabilities mu, sigma, epsilon
    if (random_action < epsilon) { // Migration
        std::swap(grid[i][j], grid[ni][nj]);
        migration++;
    } else if (random_action < epsilon + mu) {       // RPSLS Interaction selected
        if (specie != neighbour && neighbour != 0) { // Empty neighbours do not partake in RPSLS interaction
            if (dominates(specie, neighbour)) {      // Neighbour dominates, specie becomes empty
                grid[ni][nj] = 0;
            } else if (dominates(neighbour, specie)) { // Specie dominates, neighbour becomes empty
                grid[i][j] = 0;
            }
            interaction++;
        }
    } else if (random_action < epsilon + mu + sigma) { // Reproduction selected
        if (neighbour == 0) {                          // Only possible if the neighbour is empty and the specie is not
            grid[ni][nj] = specie;
            reproduction++;
        }
    } else {
        // Do nothing - step completed
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

void show() { plot_densities(steps, densityRock, densityPaper, densityScissors, densityLizard, densitySpock); }