#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

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


// draw samples from an exponential with a specified rate
double random_sample_exponential(
    const   double     rate
    ) 
{
    // initialized only once the first time the function is called
    static std::mt19937 gen(std::random_device{}());
    
    // generate random number from exponential dist.
    std::exponential_distribution<double> dist(rate);
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

    const   np_vec_f64      in_birthrate_updates_par,   // parameters used to update dynamic birth rates
    const   long            n_birthrate_updates         // number of birth rate reactions which must be updated
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
            double          t               = time_points[0];

            double          birth_rate;                         // birthrate
            vec_double      prev_state(n_birthrate_updates, -1);// previous state of the nodes with birth rate updates
            
            long            reaction_index;
            vec_double      react_propen(n_reactions);
            double          propensity_sum  = 0.0;

        
    // actual simulation part //
    for (int i = 0; i < n_time_points; ++i) {

        // while the next time point is reached
        while (t < time_points[i]) {
            
            // update birth rates in all nodes with active population size control
            for (int j = 0; j < n_birthrate_updates; j+=2) {
                // but only if the population sizes changed in the previous iteration
                if (sys_state[j] != prev_state[j] or sys_state[j+1] != prev_state[j+1]) {
                    
                    birth_rate = mu + c_b*(nss-sys_state[j]-(delta*sys_state[j+1]));
                    if (birth_rate < 0) // check for negative rates (possible if e.p.s. > nss)
                        birth_rate = 0; 
                    react_rates[j] = react_rates[j+1] = birth_rate; // set corresponding birth rates for the nodes
                }
            }
            
            // record the current state of the compartments with active population size control
            std::copy(sys_state.begin(), sys_state.begin() + n_birthrate_updates, prev_state.begin());

            propensity_sum = 0.0;
            for (int j = 0; j < n_reactions; j++) {
                // calculate global reaction propensity by multiplyin per capita rates with the number of reactants
                react_propen[j] = react_rates[j]*sys_state[state_index[j]];
                // keep track of the sum of global propensities
                propensity_sum += react_propen[j];
            }

            // if there exist any reactions (e.g. all reaction rates > 0, all reactants present at > 0)
            if (propensity_sum != 0.0) {
                // get the reaction
                reaction_index = random_event_pmf(react_propen);  
            
                // apply the reaction to the state of the system
                for (int j = 0; j < n_pops; j++) {
                    sys_state[j] += reactions(reaction_index,j);          
                }
                
                // increment time forward
                t += random_sample_exponential(propensity_sum);
            
            // if no reactions occur with probability > 0 (the system is empty), the state of the system will never change
            } else {
                t += 0.1;
            }
        }

        // write the current state of the system to the output array
        for (int j = 0; j < n_pops; ++j) {
            sys_state_sample(i,j) = sys_state[j];
        }
    }
}


// pybind 
PYBIND11_MODULE(libgillespie, m)
{
    m.def("simulate", &simulate, "simulate using gillespie");
}