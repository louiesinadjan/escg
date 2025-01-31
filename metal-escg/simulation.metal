#include <metal_stdlib>
using namespace metal;

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

kernel void step(constant uint &specie [[ buffer(0) ]],
                    constant uint &neighbour [[ buffer(1) ]],
                    constant float &random_action [[ buffer(2) ]],
                    uint id [[ thread_position_in_grid ]]) {
    thread thr; 

    // RPSLS Interaction, Reproduction, Migration actions with probabilities mu, sigma, epsilon
    if (random_action == 2) { // Migration selected
        int temp = specie; // Swap specie and neighbour
        specie = neighbour;
        neighbour = temp;
    }   else if(specie != 0){
        if(random_action == 0){ // RPSLS Interaction selected
            if (specie != neighbour && neighbour != 0) { // Empty neighbours do not partake in RPSLS interaction
                if (dominates(specie, neighbour)) {      // Neighbour dominates, specie becomes empty
                    neighbour = 0;
                } else if (dominates(neighbour, specie)) { // Specie dominates, neighbour becomes empty
                    specie = 0;
                }
            }
        } else if (random_action == 1) { // Reproduction selected
            if (neighbour == 0) { // Only possible if the neighbour is empty and the specie is not
                neighbour = specie;
            }
        }
    }
    
}

