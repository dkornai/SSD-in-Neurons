#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

typedef double f8;
typedef long int i8;

// draw samples from an exponential with mean x
double rand_exp(f8 x) {
    f8 u = rand() / (RAND_MAX + 1.0);
    return -x * log(1.0 - u);
}

// function to sample an index from an array of probabilities associated with reactions
//
//      The array represents a prob. mass function, with the index of a given
//      element being the index of the outcome, and the value of the element 
//      being the probability of that outcome. 
//      
i8 sample_discrete(f8* reaction_prob_arr_norm)
{
    f8 q = (f8) rand() / RAND_MAX; // random value between 0 and 1
    i8 i = 0;
    f8 p_sum = 0.0;
    while (p_sum < q)
    {
        p_sum += reaction_prob_arr_norm[i];
        i++;
    }
    return i - 1;
}


// all 26 possible reactions in the system
const i8 REACTIONS[36][12] = {
//  - positions:
//   ring0   ring1  ring2  ring3  ring4  ring5
    { 1, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring0 wt birth
    { 0, 1,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring0 mt birth
    { 0, 0,  1, 0,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring1 wt birth
    { 0, 0,  0, 1,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring1 mt birth
    { 0, 0,  0, 0,  1, 0,  0, 0,  0, 0,  0, 0 }, // ring2 wt birth
    { 0, 0,  0, 0,  0, 1,  0, 0,  0, 0,  0, 0 }, // ring2 mt birth
    { 0, 0,  0, 0,  0, 0,  1, 0,  0, 0,  0, 0 }, // ring3 wt birth
    { 0, 0,  0, 0,  0, 0,  0, 1,  0, 0,  0, 0 }, // ring3 mt birth
    { 0, 0,  0, 0,  0, 0,  0, 0,  1, 0,  0, 0 }, // ring4 wt birth
    { 0, 0,  0, 0,  0, 0,  0, 0,  0, 1,  0, 0 }, // ring4 mt birth
    { 0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  1, 0 }, // ring5 wt birth
    { 0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 1 }, // ring5 mt birth

    {-1, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring0 wt death
    { 0,-1,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring0 mt death
    { 0, 0, -1, 0,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring1 wt death
    { 0, 0,  0,-1,  0, 0,  0, 0,  0, 0,  0, 0 }, // ring1 mt death
    { 0, 0,  0, 0, -1, 0,  0, 0,  0, 0,  0, 0 }, // ring2 wt death
    { 0, 0,  0, 0,  0,-1,  0, 0,  0, 0,  0, 0 }, // ring2 mt death
    { 0, 0,  0, 0,  0, 0, -1, 0,  0, 0,  0, 0 }, // ring3 wt death
    { 0, 0,  0, 0,  0, 0,  0,-1,  0, 0,  0, 0 }, // ring3 mt death
    { 0, 0,  0, 0,  0, 0,  0, 0, -1, 0,  0, 0 }, // ring4 wt death
    { 0, 0,  0, 0,  0, 0,  0, 0,  0,-1,  0, 0 }, // ring4 mt death
    { 0, 0,  0, 0,  0, 0,  0, 0,  0, 0, -1, 0 }, // ring5 wt death
    { 0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0,-1 }, // ring5 mt death

    {-1, 0,  1, 0,  0, 0,  0, 0,  0, 0,  0, 0 }, //  transport wt from ring0 to ring1 
    { 0,-1,  0, 1,  0, 0,  0, 0,  0, 0,  0, 0 }, //  transport mt from ring0 to ring1
    { 0, 0, -1, 0,  1, 0,  0, 0,  0, 0,  0, 0 }, //  transport wt from ring1 to ring2
    { 0, 0,  0,-1,  0, 1,  0, 0,  0, 0,  0, 0 }, //  transport mt from ring1 to ring2
    { 0, 0,  0, 0, -1, 0,  1, 0,  0, 0,  0, 0 }, //  transport wt from ring2 to ring3
    { 0, 0,  0, 0,  0,-1,  0, 1,  0, 0,  0, 0 }, //  transport mt from ring2 to ring3
    { 0, 0,  0, 0,  0, 0, -1, 0,  1, 0,  0, 0 }, //  transport wt from ring3 to ring4
    { 0, 0,  0, 0,  0, 0,  0,-1,  0, 1,  0, 0 }, //  transport mt from ring3 to ring4
    { 0, 0,  0, 0,  0, 0,  0, 0, -1, 0,  1, 0 }, //  transport wt from ring4 to ring5
    { 0, 0,  0, 0,  0, 0,  0, 0,  0,-1,  0, 1 }, //  transport mt from ring4 to ring5
    { 1, 0,  0, 0,  0, 0,  0, 0,  0, 0, -1, 0 }, //  transport wt from ring5 to ring0
    { 0, 1,  0, 0,  0, 0,  0, 0,  0, 0,  0,-1 }, //  transport mt from ring5 to ring0
    };

// number of reactions
const i8 N_REACTIONS = 36;

// number of populations
const i8 N_POPS = 12;

// for each reaction in REACTIONS, the index of grid unit where the number of molecules determines the global reaction rate
const i8 UPDATE_INDEX[36] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, // birth reaction  
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, // death reaction
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11  // transport reaction
    };

// MAIN GILLESPIE FUNCTION
void gillespie_ring_loop(
    const f8    mu,     // death rate
    const f8    gm,     // transport rate (gamma)
    const f8    delta,  // mutant deficiency ratio
    const f8    c_b,    //ring0 birth rate control constant
    const i8    nss_s,  // carrying capacity ofring0

    const f8*   time_points,       // array of time points where the state of the system should be recorded.
    const i8    n_time_points,     // number of time points (length of 'time_points')

          i8*   sys_state,         // array holding the current state of the system (number of molecules in each location)
          i8**  sys_state_sample   // The main output. A sample of the system state at each time point specified in 'time_points'
    )
{
    // SETUP VARIABLES
    // reaction rates for each of the 26 reactions
    f8  reaction_rate_arr[36] = {
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   // birth rates, will be updated
        mu, mu, mu, mu, mu, mu, mu, mu, mu, mu, mu, mu,  // death rates, constant
        gm, gm, gm, gm, gm, gm, gm, gm, gm, gm, gm, gm   // transport rates, constant
        }; 

    f8  birth_rate;
    i8  index;

    f8* reaction_propen_arr = (f8 *) malloc(N_REACTIONS * sizeof(f8));   // unnormalizaed reaction propensity, will be overwritten during each iteration
    f8* reaction_probal_arr = (f8 *) malloc(N_REACTIONS * sizeof(f8));   // normalized reaction propensity, will be overwritten during each iteration
    f8  prop_sum = 0.0;                                                  // sum of reaction propensities, will be overwritten during each iteration

    // set t to the starting time
    f8  t = time_points[0];

    // loop through all timepoints
    for (int i = 0; i < n_time_points; ++i)
    {
        // while the absolute amount of time passed has not reached the next timepoint
        while (t < time_points[i])
        {
            // UPDATE RATES
            // update the probabilities corresponding to birth events (only these change from one iteration to the next)
            for (int j = 0; j < N_POPS; j += 2)
            {   
                birth_rate = (mu)+c_b*(nss_s-sys_state[j]-(delta*sys_state[j+1]));
                if (birth_rate < 0) 
                    birth_rate = 0; // check for negative rate
                reaction_rate_arr[j]   = birth_rate;
                reaction_rate_arr[j+1] = birth_rate;
            }

            // GET PROBABILITY OF EACH REACTION
            // multiply per capita rates with number of molecules involved, which gives the propensity
            prop_sum = 0.0;
            for (int j = 0; j < N_REACTIONS; j++)
            {
                reaction_propen_arr[j] = reaction_rate_arr[j]*sys_state[UPDATE_INDEX[j]];
                prop_sum += reaction_propen_arr[j]; // add to sum of propensities
            }
            // normalize propensities such that their sum == 1, which now gives reaction probabilities
            for (int j = 0; j < N_REACTIONS; j++)
            {
                reaction_probal_arr[j] = reaction_propen_arr[j]/prop_sum;
            }
            
            // DRAW REACTION AND INCREMENT TIME
            // select the reaction based on the array of probabilities
            index = sample_discrete(reaction_probal_arr);  
            
            // apply the changes of the reaction to the system
            for (int j = 0; j < N_POPS; j++)
            {
                sys_state[j] += REACTIONS[index][j];          
            }

            // increment time in accorance with the overall propensity
            t += rand_exp(1/prop_sum);
        }

        // update the sample of the sys state if a sample time point has been reached
        for (int j = 0; j < N_POPS; ++j)
        {
            sys_state_sample[i][j] = sys_state[j];
        }

    }

    free(reaction_probal_arr);
    free(reaction_propen_arr);
}