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

// MAIN GILLESPIE FUNCTION
void simulate(
    const f8*   time_points,       // array of time points where the state of the system should be recorded.
    const i8    n_time_points,     // number of time points (length of 'time_points')

    const i8    n_pops,
          i8*   sys_state,         // array holding the current state of the system (number of molecules in each location)
          i8**  sys_state_sample,   // The main output. A sample of the system state at each time point specified in 'time_points'

    const i8    n_reactions,
    const i8**  reactions,
          f8*   reaction_rates,
    const i8*   state_index,

    const i8    n_birthrate_updates,
    const f8**  birthrate_updates_par,
    const i8*   birthrate_updates_reaction,
    const i8*   birthrate_state_index
    )
{
    // SETUP VARIABLES
    f8  birth_rate;
    i8  reaction_index;

    f8  mu;
    f8  delta;
    f8  nss;
    f8  c_b;

    f8* reaction_propen_arr = (f8 *) malloc(n_reactions * sizeof(f8));   // unnormalizaed reaction propensity, will be overwritten during each iteration
    f8* reaction_probal_arr = (f8 *) malloc(n_reactions * sizeof(f8));   // normalized reaction propensity, will be overwritten during each iteration
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
            //     BIRTH RATES
            for (int j = 0; j < n_birthrate_updates; j += 1)
            {   
                c_b = birthrate_updates_par[j][0];
                mu = birthrate_updates_par[j][1];
                nss = birthrate_updates_par[j][2];
                delta = birthrate_updates_par[j][3];

                birth_rate = mu + c_b*(nss-sys_state[birthrate_state_index[j]]-(delta*sys_state[birthrate_state_index[j]+1]));
                if (birth_rate < 0) 
                    birth_rate = 0; // check for negative rate
                reaction_rates[birthrate_updates_reaction[j]  ] = birth_rate;
                reaction_rates[birthrate_updates_reaction[j]+1] = birth_rate;
            }

            // GET PROBABILITY OF EACH REACTION
            // multiply per capita rates with number of molecules involved, which gives the propensity
            prop_sum = 0.0;
            for (int j = 0; j < n_reactions; j++)
            {
                reaction_propen_arr[j] = reaction_rates[j]*sys_state[state_index[j]];
                prop_sum += reaction_propen_arr[j]; // add to sum of propensities
            }
            // normalize propensities such that their sum == 1, which now gives reaction probabilities
            if (prop_sum != 0.0) {; 
                for (int j = 0; j < n_reactions; j++)
                {
                    reaction_probal_arr[j] = reaction_propen_arr[j]/prop_sum;
                }
                
                // DRAW REACTION AND INCREMENT TIME
                // select the reaction based on the array of probabilities
                reaction_index = sample_discrete(reaction_probal_arr);  
                
                // apply the changes of the reaction to the system
                for (int j = 0; j < n_pops; j++)
                {
                    sys_state[j] += reactions[reaction_index][j];          
                }

                // increment time in accorance with the overall propensity
                t += rand_exp(1/prop_sum);
            
            // if prop_sum is 0, there are no reactions, so the system state does not change
            } else {;
                t += 0.001;
            }
        }

        // update the sample of the sys state if a sample time point has been reached
        for (int j = 0; j < n_pops; ++j)
        {
            sys_state_sample[i][j] = sys_state[j];
        }

    }

    free(reaction_probal_arr);
    free(reaction_propen_arr);
}

// MAIN GILLESPIE FUNCTION
void simulate_onedynamic(
    const f8*   time_points,       // array of time points where the state of the system should be recorded.
    const i8    n_time_points,     // number of time points (length of 'time_points')

    const i8    n_pops,
          i8*   sys_state,         // array holding the current state of the system (number of molecules in each location)
          i8**  sys_state_sample,   // The main output. A sample of the system state at each time point specified in 'time_points'

    const i8    n_reactions,
    const i8**  reactions,
          f8*   reaction_rates,
    const i8*   state_index,

    const i8    n_birthrate_updates,
    const f8**  birthrate_updates_par,
    const i8*   birthrate_updates_reaction,
    const i8*   birthrate_state_index
    )
{
    // SETUP VARIABLES
    f8  birth_rate;
    i8  reaction_index;

    f8  mu;
    f8  delta;
    f8  nss;
    f8  c_b;

    c_b = birthrate_updates_par[0][0];
    mu = birthrate_updates_par[0][1];
    nss = birthrate_updates_par[0][2];
    delta = birthrate_updates_par[0][3];

    f8* reaction_propen_arr = (f8 *) malloc(n_reactions * sizeof(f8));   // unnormalizaed reaction propensity, will be overwritten during each iteration
    f8* reaction_probal_arr = (f8 *) malloc(n_reactions * sizeof(f8));   // normalized reaction propensity, will be overwritten during each iteration
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
            //     BIRTH RATES   

            birth_rate = mu + c_b*(nss-sys_state[0]-(delta*sys_state[1]));
            if (birth_rate < 0) 
                birth_rate = 0; // check for negative rate
            reaction_rates[0] = birth_rate;
            reaction_rates[1] = birth_rate;

            // GET PROBABILITY OF EACH REACTION
            // multiply per capita rates with number of molecules involved, which gives the propensity
            prop_sum = 0.0;
            for (int j = 0; j < n_reactions; j++)
            {
                reaction_propen_arr[j] = reaction_rates[j]*sys_state[state_index[j]];
                prop_sum += reaction_propen_arr[j]; // add to sum of propensities
            }
            // normalize propensities such that their sum == 1, which now gives reaction probabilities
            if (prop_sum != 0.0) {; 
                for (int j = 0; j < n_reactions; j++)
                {
                    reaction_probal_arr[j] = reaction_propen_arr[j]/prop_sum;
                }
                
                // DRAW REACTION AND INCREMENT TIME
                // select the reaction based on the array of probabilities
                reaction_index = sample_discrete(reaction_probal_arr);  
                
                // apply the changes of the reaction to the system
                for (int j = 0; j < n_pops; j++)
                {
                    sys_state[j] += reactions[reaction_index][j];          
                }

                // increment time in accorance with the overall propensity
                t += rand_exp(1/prop_sum);
            
            // if prop_sum is 0, there are no reactions, so the system state does not change
            } else {;
                t += 0.001;
            }
        }

        // update the sample of the sys state if a sample time point has been reached
        for (int j = 0; j < n_pops; ++j)
        {
            sys_state_sample[i][j] = sys_state[j];
        }

    }

    free(reaction_probal_arr);
    free(reaction_propen_arr);
}