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


// all 12 possible reactions in the system
const i8 REACTIONS[12][4] = {
    {1, 0, 0, 0},  // soma wt birth
    {0, 1, 0, 0},  // soma mt birth
    {0, 0, 1, 0},  // axon wt birth
    {0, 0, 0, 1},  // axon mt birth
    {-1, 0, 0, 0}, // soma wt death
    {0, -1, 0, 0}, // soma mt death
    {0, 0, -1, 0}, // axon wt death
    {0, 0, 0, -1}, // axon mt death
    {-1, 0, 1, 0}, // transport wt from soma to axon
    {0, -1, 0, 1}, // transport mt from soma to axon
    {1, 0, -1, 0}, // transport wt from axon to soma
    {0, 1, 0, -1}  // transport mt from axon to soma
    };

// number of reactions
const i8 N_REACTIONS = 12;

// number of populations
const i8 N_POPS = 4;

// index of molecules affected by each reaction
const i8 UPDATE_INDEX[12] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};



// MAIN GILLESPIE FUNCTION
void gillespie_loop(
    const f8    mu,     // death rate
    const f8    gamma,  // transport rate
    const f8    delta,  // mutant deficiency ratio
    const f8    c_b,    // soma birth rate control constant
    const f8    c_t,    // soma to axon transport rate control constant
    const i8    nss_s,  // carrying capacity of soma
    const i8    nss_a,  // carrying capacity of axon

    const f8*   time_points,       // array of time points where the state of the system should be recorded.
    const i8    n_time_points,     // number of time points (length of 'time_points')

          i8*   sys_state,         // array holding the current state of the system (number of molecules in each location)
          i8**  sys_state_sample   // The main output. A sample of the system state at each time point specified in 'time_points'
    )
{
    // SETUP VARIABLES
    // reaction rates for each of the 12 reactions
    f8  reaction_rate_arr[12] = {0, 0, 0, 0, mu, mu, mu, mu, 0, 0, gamma, gamma};

    f8  birth_rate;
    f8  trnsp_rate;

    f8* reaction_propen_arr = (f8 *) malloc(N_REACTIONS * sizeof(f8));        // unnormalizaed reaction propensity, will be overwritten during each iteration
    f8* reaction_probal_arr = (f8 *) malloc(N_REACTIONS * sizeof(f8));   // normalized reaction propensity, will be overwritten during each iteration
    f8  prop_sum = 0.0;                                                      // sum of reaction propensities, will be overwritten during each iteration

    // set t to the starting time
    f8  t = time_points[0];

    // loop through all timepoints
    for (int i = 0; i < n_time_points; ++i)
    {
        // while the absolute amount of time passed has not reached the next timepoint
        while (t < time_points[i])
        {
            // UPDATE RATES
            // birth rates in soma (applies to wt and mt equally)
            birth_rate = (mu+mu)+c_b*(nss_s-sys_state[0]-(delta*sys_state[1]));
            if (birth_rate < 0) 
                birth_rate = 0; // check for negative rate
            reaction_rate_arr[0] = birth_rate;
            reaction_rate_arr[1] = birth_rate;

            // transport rates from soma to axon (applies to wt and mt equally)
            trnsp_rate = (gamma+gamma)+c_t*(nss_a-sys_state[2]-(delta*sys_state[3]));
            if (trnsp_rate < 0)
                trnsp_rate = 0; // check for negative rate
            reaction_rate_arr[8] = trnsp_rate;
            reaction_rate_arr[9] = trnsp_rate;

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
            int index = sample_discrete(reaction_probal_arr);  
            
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