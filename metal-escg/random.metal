using namespace metal;
#include <metal_stdlib>

#define UPPER_MASK         0x80000000
#define LOWER_MASK         0x7fffffff
#define TEMPERING_MASK_B   0x9d2c5680
#define TEMPERING_MASK_C   0xefc60000
#define MATRIX_A           0x9908b0dfUL
#define STATE_VECTOR_LENGTH 624
#define STATE_VECTOR_M      397

struct MT19937 {
    uint state[STATE_VECTOR_LENGTH];
    uint index;
};

/// A simple 32-bit hash function (similar to a finalizer)
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

/// Improved seeding function that combines the seed with the thread id
void seed_mt(thread MT19937 &mt, uint seed, uint id) {
    // Mix in the thread id using a golden-ratio constant, then hash the result.
    seed ^= id * 0x9e3779b9U;
    seed = hash(seed);
    
    mt.state[0] = seed;
    for (uint i = 1; i < STATE_VECTOR_LENGTH; ++i) {
        mt.state[i] = (1812433253U * (mt.state[i - 1] ^ (mt.state[i - 1] >> 30)) + i);
    }
    mt.index = STATE_VECTOR_LENGTH;
}

/// The twist transformation as in the standard algorithm.
void twist(thread MT19937 &mt) {
    const uint mag01[2] = {0x0UL, MATRIX_A};
    uint y;
    
    for (uint i = 0; i < STATE_VECTOR_LENGTH - STATE_VECTOR_M; i++) {
        y = (mt.state[i] & UPPER_MASK) | (mt.state[i + 1] & LOWER_MASK);
        mt.state[i] = mt.state[i + STATE_VECTOR_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    
    for (uint i = STATE_VECTOR_LENGTH - STATE_VECTOR_M; i < STATE_VECTOR_LENGTH - 1; i++) {
        y = (mt.state[i] & UPPER_MASK) | (mt.state[i + 1] & LOWER_MASK);
        mt.state[i] = mt.state[i + (STATE_VECTOR_M - STATE_VECTOR_LENGTH)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    
    y = (mt.state[STATE_VECTOR_LENGTH - 1] & UPPER_MASK) | (mt.state[0] & LOWER_MASK);
    mt.state[STATE_VECTOR_LENGTH - 1] = mt.state[STATE_VECTOR_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
    
    mt.index = 0;
}

/// Extracts a number using the standard MT19937 tempering.
uint extract(thread MT19937 &mt) {
    if (mt.index >= STATE_VECTOR_LENGTH) {
        twist(mt);
    }
    
    uint y = mt.state[mt.index++];
    
    // Standard tempering steps (do not add extra mixing)
    y ^= (y >> 11);
    y ^= (y << 7) & TEMPERING_MASK_B;
    y ^= (y << 15) & TEMPERING_MASK_C;
    y ^= (y >> 18);
    
    return y;
}

/// Returns a random integer in [0, N - 1]
uint random_cell(thread MT19937 &mt, int N) {
    return extract(mt) % N;
}

/// Returns a random integer in [0, 3]
uint random_int_0_3(thread MT19937 &mt) {
    return extract(mt) % 4;
}

/// Returns a random integer in [0, 7]
uint random_int_0_7(thread MT19937 &mt) {
    return extract(mt) % 8;
}

/// Returns a random float in [0, 1]
float random_float_0_1(thread MT19937 &mt) {
    return static_cast<float>(extract(mt)) / 4294967295.0f;
}


/// Kernel for generating random cell indices
kernel void mt_random_cells(
    const device uint *seeds [[ buffer(0) ]],
    device uint *results [[ buffer(1) ]],
    constant int &N [[ buffer(2) ]], 
    uint id [[ thread_position_in_grid ]]) {
    thread MT19937 mt;
    seed_mt(mt, seeds[id], id);

    // Burn-in 
    for (uint i = 0; i < 50000; ++i) { extract(mt); }

    for (uint i = 0; i < 10000; ++i) {
        results[id * 10000 + i] = random_cell(mt, N);
    }
}

/// Kernel for generating random neighbour directions
kernel void mt_random_neighbours(
    const device uint *seeds [[ buffer(0) ]],
    device uint *results [[ buffer(1) ]],
    constant bool &moore [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]) {

    thread MT19937 mt;
    seed_mt(mt, seeds[id], id);
    // Burn-in of 1,000,000 iterations
    // Lower degrees of freedom (0-3) so more burn-in is required
    for (uint i = 0; i < 1000000; ++i) { extract(mt); }

    if(!moore){
        for (uint i = 0; i < 10000; ++i) {
            results[id * 10000 + i] = random_int_0_3(mt);
        }
    } else {
        for (uint i = 0; i < 10000; ++i) {
            results[id * 10000 + i] = random_int_0_7(mt);
        }
    }
    
}

/// Kernel for generating random floating-point numbers in [0, 1]
kernel void mt_random_actions(
    const device uint *seeds [[ buffer(0) ]],                          
    device float *results [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]) {

    thread MT19937 mt;
    seed_mt(mt, seeds[id], id);

    // Burn-in of 200,000
    for (uint i = 0; i < 50000; ++i) { extract(mt); } 

    for (uint i = 0; i < 10000; ++i) {
        results[id * 10000 + i] = random_float_0_1(mt);
    }
}