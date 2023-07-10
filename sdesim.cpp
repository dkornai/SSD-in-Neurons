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

using vec_int = std::vector<int>;
using vec_double = std::vector<double>;

typedef py::array_t<double> np_vec_f64; // 1d np.float64 array
typedef py::array_t<int> np_vec_int; // 1d np.int64 array 

// 
typedef Eigen::Ref<Eigen::MatrixXi> np_mat_int; // 2d np.int64 array 

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
vec_int get_int_vec_from_np(
    const   np_vec_int     arr
    )
{
    py::buffer_info info = arr.request();
    int* data = static_cast<int*>(info.ptr);
    std::size_t size = info.size;
    std::vector<int> vec(data, data + size);
    return vec;
}

Eigen::VectorXi get_eig_int_vec_from_np(py::array_t<int> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected a 1D numpy array");
    }

    int* ptr = static_cast<int*>(buf.ptr);
    Eigen::Map<Eigen::VectorXi> eigenVector(ptr, buf.size);

    return eigenVector;
}

// check that all wildtypes or mutant are dead
bool takeover(
    const   vec_int    vec,
    const   int         offset // offset of 0 is wt, offset of 1 is mt
    ) 
{
    for (size_t i = offset; i < vec.size(); i += 2) {
        if (vec[i] != 0) {
            return false;
        }
    }
    return true;
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
void sim_gillespie(
    const   np_vec_f64      in_time_points,             // points in time where system state should be recorded
    
    const   np_vec_int      in_sys_state,               // starting 'sys_state'
            np_mat_int     sys_state_sample,           // array where the 'sys_state' is recorded at each time in 'time_points'
            
    const   np_mat_int     reactions,                  // 'sys_state' updates corresponding to each possible reaction
    const   np_vec_f64      in_react_rates,             // per capita reaction rates
    const   np_vec_int      in_state_index,             // indeces of system state variables from which propensity is calculated

    const   np_vec_f64      in_birthrate_updates_par,   // parameters used to update dynamic birth rates
    const   int            n_birthrate_updates         // number of birth rate reactions which must be updated
    )
{
    // c++ versions of numpy arrays
    const   vec_double      time_points     = get_double_vec_from_np(in_time_points);
            vec_double      react_rates     = get_double_vec_from_np(in_react_rates);
    const   vec_int        state_index     = get_int_vec_from_np(in_state_index);
            vec_int        sys_state       = get_int_vec_from_np(in_sys_state);
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
            
            int            reaction_index;
            vec_double      react_propen(n_reactions);
            double          propensity_sum  = 0.0;

        
    // actual simulation part //
    for (int i = 0; i < n_time_points; ++i) {
        
        // if the system has all mutants, or all wildtypes, the system state will not change, so all further samples can be set to the current state
        if ((takeover(sys_state, 0) or takeover(sys_state, 1)) and t > (10 + time_points[0])) {
            // do nothing
    
        // otherwise do the actual simulation    
        } else {
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
        }
        
        // write the current state of the system to the output array
        for (int j = 0; j < n_pops; ++j) {
            sys_state_sample(i,j) = sys_state[j];
        }

        
    }
}



// for each reaction rate in the input vector, randomly generate the number of events for that reaction using a poisson distribution.
Eigen::VectorXi get_poisson_number_of_events(
    const   Eigen::VectorXd&    react_propensity
    ) 
{
    // Making generator static ensures it's initialized only once
    static std::mt19937 gen(std::random_device{}());
    
    static Eigen::VectorXi result(react_propensity.size());

    for (int i = 0; i < react_propensity.size(); ++i) {
        std::poisson_distribution<> dist(react_propensity[i]);
        result[i] = dist(gen);
    }

    return result;
}

// get the change in the system state due to the number of events (n_events_per_reaction_type[i]) events occuring for each reaction (reactions[i])
Eigen::VectorXi get_system_state_change(
    const   Eigen::VectorXi&    n_events_per_reaction, 
    const   Eigen::MatrixXi&    reactions
    ) 
{
    int n = n_events_per_reaction.size();

    // Create a static diagonal matrix that retains its value across function calls
    static Eigen::DiagonalMatrix<int, Eigen::Dynamic> diagonalized_vector(n);
    diagonalized_vector.diagonal() = n_events_per_reaction;

    // Multiply the diagonalized_vector by the 'reactions' matrix to get the change in the system change bought about by a set number of each reaction type
    Eigen::MatrixXi product = diagonalized_vector * reactions;

    // Perform a column-wise sum of the resulting matrix to get the over all change in the system state
    Eigen::VectorXi column_sum = product.colwise().sum();

    return column_sum;
}


// TAU LEAPING FUNCTION //
void sim_tauleaping(
    const   np_vec_f64      in_time_points,             // points in time where system state should be recorded
    
    const   np_vec_int      in_sys_state,               // starting 'sys_state'
            np_mat_int     sys_state_sample,           // array where the 'sys_state' is recorded at each time in 'time_points'
            
    const   np_mat_int     reactions,                  // 'sys_state' updates corresponding to each possible reaction
    const   np_vec_f64      in_react_rates,             // per capita reaction rates
    const   np_vec_int      in_state_index,             // indeces of system state variables from which propensity is calculated

    const   np_vec_f64      in_birthrate_updates_par,   // parameters used to update dynamic birth rates
    const   int            n_birthrate_updates         // number of birth rate reactions which must be updated
    )
{
    // c++ versions of numpy arrays
    const   vec_double      time_points     = get_double_vec_from_np(in_time_points);
            vec_double      percap_r_rates  = get_double_vec_from_np(in_react_rates);
    const   vec_int         state_index     = get_int_vec_from_np(in_state_index);
            Eigen::VectorXi sys_state       = get_eig_int_vec_from_np(in_sys_state);
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
            double          timestep        = 0.01;

            double          birth_rate;                         // birthrate
            
            Eigen::VectorXd global_react_rates(n_reactions);    // propensity of each reaction
            Eigen::VectorXi n_events_per_reaction(n_reactions); // number of times each reaction occurs
            Eigen::VectorXi state_change(n_pops);               // change in the system state from one time step to the next

        
    // actual simulation part //
    for (int i = 0; i < n_time_points; ++i) {
        
        // while the next time point is reached
        while (t < time_points[i]) {
            
            // update birth rates in all nodes with active population size control
            for (int j = 0; j < n_birthrate_updates; j+=2) {
                birth_rate = mu + c_b*(nss-sys_state[j]-(delta*sys_state[j+1]));
                
                if (birth_rate < 0) // check for negative rates (possible if e.p.s. > nss)
                    birth_rate = 0; 
                
                percap_r_rates[j] = percap_r_rates[j+1] = birth_rate; // set corresponding birth rates for the nodes
            }

            // calculate global reaction propensity by multiplyin per capita rates with the number of reactants
            for (int j = 0; j < n_reactions; j++) {
                global_react_rates[j] = percap_r_rates[j]*sys_state[state_index[j]]*timestep;
            }

            // given the reaction rates, get the number of times each reaction occurs during the timestep
            n_events_per_reaction = get_poisson_number_of_events(global_react_rates);

            // get the change in the system state corresponding to the number of each reactions
            state_change = get_system_state_change(n_events_per_reaction, reactions);

            // update the state of the system
            sys_state += state_change;
            
            // increment time forward
            t += timestep;
        }
        
        // write the current state of the system to the output array
        for (int j = 0; j < n_pops; ++j) {
            sys_state_sample(i,j) = sys_state[j];
        }

        
    }
}


// PYBIND // 
PYBIND11_MODULE(libsdesim, m)
{
    m.def("sim_gillespie", &sim_gillespie, "simulate using gillespie");
    m.def("sim_tauleaping", &sim_tauleaping, "simulate using tau leaping");
}