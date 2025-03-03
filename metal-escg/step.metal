#include <metal_stdlib>
using namespace metal;

 // Species dominates neighbour? 
bool dominates(int specie, int neighbour) {
    switch (specie) {
        case 1: // ROCK (crushes 3: SCISSORS, crushes 4: LIZARD)
            return neighbour == 4; // Absence of Rock - Scissors interaction
            // return (neighbour == 3 || neighbour == 4);
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

int action(float action_prob, float mu, float sigma) {
    if (action_prob < mu) {
        return 1; // Interaction
    } else if (action_prob < mu + sigma) {
        return 2; // Reproduction
    } else {
        return 3; // Migration
    }
}

// cells[i] will be processed with neighbour_dirs[i] and action_probabilities[i]
kernel void step(
    const device int *cells [[ buffer(0) ]], 
    const device int *neighbour_dirs [[ buffer(1) ]],
    const device float *action_probabilities [[ buffer(2) ]],
    constant float &mu [[ buffer(3) ]], 
    constant float &sigma [[ buffer(4) ]], 
    constant int &L [[ buffer(5) ]], 
    constant int &H [[ buffer(6) ]], 
    device atomic_int *grid [[ buffer(7) ]],
    uint id [[ thread_position_in_grid ]]) {

    // Precompute the offsets for each direction
    const int N = L * H;

    const int offsets[8][2] = {
        {-1, 0}, // Up
        {1, 0},  // Down
        {0, -1}, // Left
        {0, 1},   // Right
        {-1, -1}, // Up-Left
        {-1, 1}, // Up-Right 
        {1, -1}, // Down-Left
        {1, 1} // Down-Right
    };

    int cellsPerThread = N / 1000;

    for (int i = 0; i < cellsPerThread; i++) {

        // Get the flattened index of the current cell.
        int cell_index = cells[id * cellsPerThread + i];

        if (cell_index == -1 || cell_index >= N) {
            return;
        }

        // Read the species from the grid atomically
        int specie = atomic_load_explicit(&grid[cell_index], memory_order_relaxed);
        
        // Compute the action based on the action probability.
        // (Assume an 'action' function exists that returns an int (e.g. 1,2,3))
        int act = action(action_probabilities[id * cellsPerThread + i], mu, sigma);
        
        // Get the neighbor direction:
        // 0 = up, 1 = down, 2 = left, 3 = right.
        int n_dir = neighbour_dirs[id * cellsPerThread + i];
        
        // Convert the 1D cell index to 2D coordinates.
        int row = cell_index / L;
        int col = cell_index % L;
        
        // Compute the new row and column based on the neighbor direction with wrapping.
        int n_row = (row + offsets[n_dir][0] + H) % H;  // Wrap within height H
        int n_col = (col + offsets[n_dir][1] + L) % L;
        
        // Convert the new row and column back into a 1D index.
        int neighbour_index = n_row * L + n_col;
        int neighbour_specie = atomic_load_explicit(&grid[neighbour_index], memory_order_relaxed);
        
        if (act == 1) { // Interaction: apply the dominance rules
            if (dominates(specie, neighbour_specie)) {
                atomic_store_explicit(&grid[neighbour_index], 0, memory_order_relaxed); // Remove neighbour
            } else if (dominates(neighbour_specie, specie)) {
                atomic_store_explicit(&grid[cell_index], 0, memory_order_relaxed); // Remove self
            }
        } else if (act == 2) { // Reproduction: if one of the cells is empty, copy the nonempty specie
            if (neighbour_specie == 0 && specie != 0) {
                atomic_store_explicit(&grid[neighbour_index], specie, memory_order_relaxed);
            } else if (specie == 0 && neighbour_specie != 0) {
                atomic_store_explicit(&grid[cell_index], neighbour_specie, memory_order_relaxed);
            }
        } else if (act == 3) { // Migration: swap the species of the current cell and the neighbour cell
            atomic_exchange_explicit(&grid[cell_index], neighbour_specie, memory_order_relaxed);
            atomic_exchange_explicit(&grid[neighbour_index], specie, memory_order_relaxed);
        }
    }
}