#include <metal_stdlib>
using namespace metal;

struct MT19937 {
    uint state[624];
    uint index;
};

void initialise(thread MT19937 &mt, uint seed) {
    mt.state[0] = seed;
    for (uint i = 1; i < 624; ++i) {
        mt.state[i] = 1812433253 * (mt.state[i - 1] ^ (mt.state[i - 1] >> 30)) + i;
    }
    mt.index = 624;
}

void twist(thread MT19937 &mt) {
    for (uint i = 0; i < 624; ++i) {
        uint y = (mt.state[i] & 0x80000000) + (mt.state[(i + 1) % 624] & 0x7fffffff);
        mt.state[i] = mt.state[(i + 397) % 624] ^ (y >> 1);
        if (y % 2 != 0) {
            mt.state[i] ^= 2567483615;
        }
    }
    mt.index = 0;
}

uint extract(thread MT19937 &mt) {
    if (mt.index >= 624) {
        twist(mt);
    }
    uint y = mt.state[mt.index++];
    y ^= y >> 11;
    y ^= (y << 7) & 2636928640;
    y ^= (y << 15) & 4022730752;
    y ^= y >> 18;
    return y;
}

kernel void mt_random_cells(const device uint *seeds [[ buffer(0) ]],
                             device uint *results [[ buffer(1) ]],
                             uint id [[ thread_position_in_grid ]]) {
    thread MT19937 mt; // Each thread has its own MT19937 state
    initialise(mt, seeds[id]); // Initialise the state with the seed

    for (uint i = 0; i < 10000; ++i) { // Generate 1000 random numbers per thread
        results[id * 10000 + i] = extract(mt) % 40000; // Random number between 0 and 39,999
    }
}

kernel void mt_random_neighbours(const device uint *seeds [[ buffer(0) ]],
                             device uint *results [[ buffer(1) ]],
                             uint id [[ thread_position_in_grid ]]) {
    thread MT19937 mt; // Each thread has its own MT19937 state
    initialise(mt, seeds[id]); // Initialise the state with the seed

    for (uint i = 0; i < 10000; ++i) { // Generate 1000 random numbers per thread
        results[id * 10000 + i] = extract(mt) % 4; // Random number between 0 and 3
    }
}

kernel void mt_random_actions(const device uint *seeds [[ buffer(0) ]],
                             device float *results [[ buffer(1) ]],
                             uint id [[ thread_position_in_grid ]]) {
    thread MT19937 mt; // Each thread has its own MT19937 state
    initialise(mt, seeds[id]); // Initialise the state with the seed

    for (uint i = 0; i < 10000; ++i) { // Generate 1000 random floats per thread
        results[id * 10000 + i] = static_cast<float>(extract(mt)) / 4294967295.0f; // Normalise to [0, 1]
    }
}