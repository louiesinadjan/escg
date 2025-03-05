#include <metal_stdlib>
using namespace metal;

kernel void rmf_compute_densities(
    const device int *grid [[ buffer(0) ]],
    device atomic_int *densityResults [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]) {
    int species = grid[id];
    switch (species) {
        case 0:
            atomic_fetch_add_explicit(&densityResults[0], 1, memory_order_relaxed);
            break;
        case 1:
            atomic_fetch_add_explicit(&densityResults[1], 1, memory_order_relaxed);
            break;
        case 2:
            atomic_fetch_add_explicit(&densityResults[2], 1, memory_order_relaxed);
            break;
        case 3:
            atomic_fetch_add_explicit(&densityResults[3], 1, memory_order_relaxed);
            break;
        case 4:
            atomic_fetch_add_explicit(&densityResults[4], 1, memory_order_relaxed);
            break;
        case 5:
            atomic_fetch_add_explicit(&densityResults[5], 1, memory_order_relaxed);
            break;
        default:
            // If an unexpected value appears, do nothing.
            break;
    }
}
