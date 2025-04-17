#include <metal_stdlib>
using namespace metal;

#define STATE_VECTOR_LENGTH 624
#define STATE_VECTOR_M      397
#define MATRIX_A            0x9908b0dfU
#define UPPER_MASK          0x80000000U
#define LOWER_MASK          0x7fffffffU
#define TEMPERING_MASK_B    0x9d2c5680U
#define TEMPERING_MASK_C    0xefc60000U
#define MT_INIT_MULTIPLIER  1812433253U

struct MT19937 {
    uint state[STATE_VECTOR_LENGTH];
    uint index;
};

void initialise(thread MT19937 &mt, uint seed) {
    mt.state[0] = seed;
    for (uint i = 1; i < STATE_VECTOR_LENGTH; ++i) {
        mt.state[i] = MT_INIT_MULTIPLIER * (mt.state[i - 1] ^ (mt.state[i - 1] >> 30)) + i;
    }
    mt.index = STATE_VECTOR_LENGTH;
}

void twist(thread MT19937 &mt) {
    for (uint i = 0; i < STATE_VECTOR_LENGTH; ++i) {
        uint y = (mt.state[i] & UPPER_MASK) + (mt.state[(i + 1) % STATE_VECTOR_LENGTH] & LOWER_MASK);
        mt.state[i] = mt.state[(i + STATE_VECTOR_M) % STATE_VECTOR_LENGTH] ^ (y >> 1);
        if (y % 2 != 0) {
            mt.state[i] ^= MATRIX_A;
        }
    }
    mt.index = 0;
}

uint extract(thread MT19937 &mt) {
    if (mt.index >= STATE_VECTOR_LENGTH) {
        twist(mt);
    }
    uint y = mt.state[mt.index++];
    y ^= y >> 11;
    y ^= (y << 7) & TEMPERING_MASK_B;
    y ^= (y << 15) & TEMPERING_MASK_C;
    y ^= y >> 18;
    return y;
}

kernel void mersenne_twister(const device uint *seeds [[ buffer(0) ]],
                             device uint *results [[ buffer(1) ]],
                             uint id [[ thread_position_in_grid ]]) {
    thread MT19937 mt; // Each thread has its own MT19937 state
    initialise(mt, seeds[id]); // Initialise the state with the seed
    for (uint i = 0; i < 50000; ++i) { // Generate 500 random numbers per thread
        results[id * 500000 + i] = extract(mt);
    }
}

