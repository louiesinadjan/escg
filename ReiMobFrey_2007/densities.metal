#include <metal_stdlib>
using namespace metal;

kernel void compute_densities(
    device const int*       grid          [[ buffer(0) ]],  // Flattened grid array (40000 elements)
    device atomic_int*      result        [[ buffer(1) ]],  // Global output: 6 atomic ints
    uint                    id            [[ thread_position_in_grid ]]
) {
    int species = grid[id];
    switch (species) {
        case 0:
            atomic_fetch_add_explicit(&result[0], 1, memory_order_relaxed);
            break;
        case 1:
            atomic_fetch_add_explicit(&result[1], 1, memory_order_relaxed);
            break;
        case 2:
            atomic_fetch_add_explicit(&result[2], 1, memory_order_relaxed);
            break;
        case 3:
            atomic_fetch_add_explicit(&result[3], 1, memory_order_relaxed);
            break;
        case 4:
            atomic_fetch_add_explicit(&result[4], 1, memory_order_relaxed);
            break;
        case 5:
            atomic_fetch_add_explicit(&result[5], 1, memory_order_relaxed);
            break;
        default:
            // If an unexpected value appears, do nothing.
            break;
    }
}