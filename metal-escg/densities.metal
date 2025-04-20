#include <metal_stdlib>
using namespace metal;

kernel void compute_densities(
    device const int*       grid          [[ buffer(0) ]],  // Flattened grid array (L * H cells)
    device atomic_int*      result        [[ buffer(1) ]],  
    uint                    id            [[ thread_position_in_grid ]]
) {
    int species = grid[id];
    if(species < 0){
        return;
    }
    atomic_fetch_add_explicit(&result[species], 1, memory_order_relaxed);
}