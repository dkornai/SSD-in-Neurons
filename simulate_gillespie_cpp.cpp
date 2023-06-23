#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <iostream>
#include <string> 

namespace py = pybind11;


// CUSTOM TYPES //

using vec_long = std::vector<long>;
using vec_double = std::vector<double>;

typedef py::array_t<double> np_vec_f64; // 1d np.float64 array
typedef py::array_t<long> np_vec_i64; // 1d np.int64 array 

typedef Eigen::Matrix<long, -1, -1> MatrixXl; // matrix with long entries
typedef Eigen::Ref<MatrixXl> np_mat_long; // 2d np.int64 array 


// HELPER FUNCTIONS //

// c++ vector from np.float64 array
vec_double get_double_vec_from_np(
    const   np_vec_f64   arr
    )
{
    py::buffer_info info = arr.request();
    double* data = static_cast<double*>(info.ptr);
    std::size_t size = info.size;
    std::vector<double> vec(data, data + size);
    return vec;
}

// c++ vector from np.int64 array
vec_long get_long_vec_from_np(
    const   np_vec_i64     arr
    )
{
    py::buffer_info info = arr.request();
    long* data = static_cast<long*>(info.ptr);
    std::size_t size = info.size;
    std::vector<long> vec(data, data + size);
    return vec;
}


// draw samples from an exponential with a specified mean 
double random_sample_exponential(
    const   double     mean
    ) 
{
    // initialized only once the first time the function is called
    static std::mt19937 gen(std::random_device{}());
    
    // generate random number from exponential dist.
    std::exponential_distribution<double> dist(1.0 / mean);
    return dist(gen);
}


// function to sample an index from an array of probabilities associated with reactions
int random_event_pmf(
    const   vec_double& react_probal
    ) 
{
    // initialized only once the first time the function is called
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // sample from the p.m.f. defined by 'react_probal'
    std::discrete_distribution<> dist(react_probal.begin(), react_probal.end());
    return dist(gen);
}


// MAIN GILLESPIE FUNCTION //
void simulate(
    const   np_vec_f64      in_time_points,             // points in time where system state should be recorded
    
    const   np_vec_i64      in_sys_state,               // starting 'sys_state'
            np_mat_long     sys_state_sample,           // array where the 'sys_state' is recorded at each time in 'time_points'
            
    const   np_mat_long     reactions,                  // 'sys_state' updates corresponding to each possible reaction
    const   np_vec_f64      in_react_rates,             // per capita reaction rates
    const   np_vec_i64      in_state_index,             // indeces of system state variables from which propensity is calculated
    const   np_vec_f64      in_birthrate_updates_par    // parameters used to update dynamic birth rates
    )
{
    // c++ versions of numpy arrays
    const   vec_double      time_points     = get_double_vec_from_np(in_time_points);
            vec_double      react_rates     = get_double_vec_from_np(in_react_rates);
    const   vec_long        state_index     = get_long_vec_from_np(in_state_index);
            vec_long        sys_state       = get_long_vec_from_np(in_sys_state);
    const   vec_double      br_up_par       = get_double_vec_from_np(in_birthrate_updates_par);

    // counts often accessed during loops
    const   int             n_time_points   = time_points.size();
    const   int             n_pops          = sys_state.size();
    const   int             n_reactions     = state_index.size();

    // variables used in calculating the dynamic birth rate in compartment 0
    const   double          c_b             = br_up_par[0];
    const   double          mu              = br_up_par[1];
    const   double          nss             = br_up_par[2];
    const   double          delta           = br_up_par[3];
    
    // variables rewritten throughout simulation
            double          birth_rate;
            long            reaction_index;

            vec_double      react_propen(n_reactions);
            vec_double      react_probal(n_reactions);

            double          propensity_sum  = 0.0;
            double          t               = time_points[0];


    // actual simulation part //
    for (int i = 0; i < n_time_points; ++i) {

        // while the next time point is reached
        while (t < time_points[i]) {
            
            // update birth rates in compartment 0 (main node)
            birth_rate = mu + c_b*(nss-sys_state[0]-(delta*sys_state[1]));
            if (birth_rate < 0) // check for negative rates (possible if e.p.s. > nss)
                birth_rate = 0; 
            react_rates[0] = react_rates[1] = birth_rate;

            propensity_sum = 0.0;
            for (int j = 0; j < n_reactions; j++) {
                // calculate global reaction propensity by multiplyin per capita rates with the number of reactants
                react_propen[j] = react_rates[j]*sys_state[state_index[j]];
                // keep track of the sum of global propensities
                propensity_sum += react_propen[j];
            }

            // if there exist any reactions (e.g. all reaction rates > 0, all reactants present at > 0)
            if (propensity_sum != 0.0) {
                // normalize global propensities such that their sum is 1 (forming a pmf)
                for (int j = 0; j < n_reactions; j++) {
                    react_probal[j] = react_propen[j]/propensity_sum;
                }
                
                // randomly generate a reaction
                reaction_index = random_event_pmf(react_probal);  
            
                // apply the reaction to the state of the system
                for (int j = 0; j < n_pops; j++) {
                    sys_state[j] += reactions(reaction_index,j);          
                }
                
                // increment time forward
                t += random_sample_exponential(1/propensity_sum);
            
            // if no reactions occur with probability > 0 (the system is empty), the state of the system will never change
            } else {
                t += 0.001;
            }
        }

        // write the current state of the system to the output array
        for (int j = 0; j < n_pops; ++j) {
            sys_state_sample(i,j) = sys_state[j];
        }
    }
}



PYBIND11_MODULE(libgillespie, m)
{
    m.def("simulate", &simulate, "simulate using gillespie");
}