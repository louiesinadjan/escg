#include <metal_stdlib>
using namespace metal;

inline float dominates(int specie, int neighbour, int speciesNum, constant float* dominance) {
    return dominance[(specie * speciesNum + neighbour)];
}

// cells[i] will be processed with neighbour_dirs[i] and invasion_probabilities[i]
kernel void step(
    const device int *cells [[ buffer(0) ]], 
    const device int *neighbour_dirs [[ buffer(1) ]],
    const device float *invasion_probabilities [[ buffer(2) ]],

    constant float *dominance [[ buffer(3) ]], 
    constant float &beta [[ buffer(4) ]], 
    constant float &gamma [[ buffer(5) ]], 

    constant int &L [[ buffer(6) ]], 
    constant int &H [[ buffer(7) ]],
    constant bool &flux [[ buffer(8) ]],
    constant int &speciesNum [[ buffer(9) ]],
    constant int &numRandoms [[ buffer(10) ]],
    constant bool &maxStep [[ buffer(11) ]],
    device atomic_int *grid [[ buffer(12) ]],


    uint id [[ thread_position_in_grid ]]) {

    const int N = L * H;

    // Precompute the offsets for each direction
    const int offsets[8][2] = {
        {-1, 0},    // Up
        {1, 0},     // Down
        {0, -1},    // Left
        {0, 1},     // Right
        {-1, -1},   // Up-Left
        {-1, 1},    // Up-Right 
        {1, -1},    // Down-Left
        {1, 1}      // Down-Right
    };

    int cellsPerThread = maxStep ? numRandoms / 1000 : N / 1000;

    for (int i = 0; i < cellsPerThread; i++) {

        int cell_index = cells[id * cellsPerThread + i];
        if (cell_index <= -1 || cell_index >= N) {
            return;
        }

        int specie = atomic_load_explicit(&grid[cell_index], memory_order_relaxed);

        int n_dir = neighbour_dirs[id * cellsPerThread + i];
        
        // Convert the 1D cell index to 2D coordinates.
        int row = cell_index / L;
        int col = cell_index % L;
        
        // Compute the new row and column based on the direction.
        int n_row, n_col;
        if (flux) {
            n_row = (row + offsets[n_dir][0] + H) % H;
            n_col = (col + offsets[n_dir][1] + L) % L;
        } else {
            n_row = row + offsets[n_dir][0];
            if(n_row < 0 || n_row >= H) {
                n_row = row - offsets[n_dir][0];
            }
            n_col = col + offsets[n_dir][1];
            if(n_col < 0 || n_col >= L) {
                n_col = col - offsets[n_dir][1];
            }
        }

        // Convert the new row and column back into a 1D index.
        int neighbour_index = n_row * L + n_col;
        int neighbour_specie = atomic_load_explicit(&grid[neighbour_index], memory_order_relaxed);

        if(specie == neighbour_specie){
            continue; 
        }

        float invasion_prob = invasion_probabilities[id * cellsPerThread + i];

        // Let specie be the lower specie
        if(specie > neighbour_specie){
            int temp = specie;
            specie = neighbour_specie;
            neighbour_specie = temp;

            // Swap the indices
            int temp2 = cell_index;
            cell_index = neighbour_index;
            neighbour_index = temp2;
        }

        float specieDominates = dominates(specie, neighbour_specie, speciesNum, dominance);
        float neighbourDominates = dominates(neighbour_specie, specie, speciesNum, dominance);
        if(specieDominates == 0 && neighbourDominates == 0){
            continue;
        } else if(invasion_prob < specieDominates){
            atomic_store_explicit(&grid[neighbour_index], specie, memory_order_relaxed); // Replace neighbour
        } else if(invasion_prob < neighbourDominates){
            atomic_store_explicit(&grid[cell_index], neighbour_specie, memory_order_relaxed); // Replace specie
        }

    }
}