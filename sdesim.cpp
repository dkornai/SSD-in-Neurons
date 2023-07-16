#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#define EIGEN_NO_DEBUG

namespace py = pybind11;


// CUSTOM TYPES //

using vec_int = std::vector<int>;
using vec_double = std::vector<double>;



typedef py::array_t<double> np_vec_f64; // 1d np.float64 array
typedef py::array_t<int> np_vec_i32; // 1d np.int32 array 

typedef Eigen::Ref<Eigen::MatrixXi> np_mat_i32; // 2d np.int32 array 

typedef Eigen::DiagonalMatrix<int, Eigen::Dynamic> Eigen_diag_mat; // diagonalized matrix

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
    const   np_vec_i32     arr
    )
{
    py::buffer_info info = arr.request();
    int* data = static_cast<int*>(info.ptr);
    std::size_t size = info.size;
    std::vector<int> vec(data, data + size);
    return vec;
}

// Eigen int vector from np.int32 array
Eigen::VectorXi get_eig_int_vec_from_np(py::array_t<int> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected a 1D numpy array");
    }

    int* ptr = static_cast<int*>(buf.ptr);
    Eigen::Map<Eigen::VectorXi> eigenVector(ptr, buf.size);

    return eigenVector;
}



// GILLESPIE FUNCTION //
void sim_gillespie(
    const   np_vec_f64      in_time_points,             // points in time where system state should be recorded
    
    const   np_vec_i32      in_sys_state,               // starting 'sys_state'
            np_mat_i32      sys_state_sample,           // array where the 'sys_state' is recorded at each time in 'time_points'
            
    const   np_mat_i32      reactions,                  // 'sys_state' updates corresponding to each possible reaction
    const   np_vec_f64      in_percap_r_rates,             // per capita reaction rates
    const   np_vec_i32      in_state_index,             // indeces of system state variables from which propensity is calculated

    const   np_vec_f64      in_birthrate_updates_par,   // parameters used to update dynamic birth rates
    const   int             n_birthrate_updates         // number of birth rate reactions which must be updated
    )
{
    // ### VARIABLE SETUP #### //

    // c++ versions of numpy arrays
    const   vec_double      time_points     = get_double_vec_from_np(in_time_points);
            vec_double      percap_r_rates  = get_double_vec_from_np(in_percap_r_rates);
    const   vec_int         state_index     = get_int_vec_from_np(in_state_index);
            Eigen::VectorXi sys_state       = get_eig_int_vec_from_np(in_sys_state);
   
    // counts often accessed during loops
    const   int             n_time_points   = time_points.size();
    const   int             n_reactions     = state_index.size();

    // variables used in calculating the dynamic birth rate in compartment 0
    const   vec_double      br_up_par       = get_double_vec_from_np(in_birthrate_updates_par);
    const   double          c_b             = br_up_par[0];
    const   double          mu              = br_up_par[1];
    const   double          nss             = br_up_par[2];
    const   double          delta           = br_up_par[3];
    const   double          brh1            = mu + c_b * nss;   // precomputed values that do not change across iterations
    const   double          brh2            = c_b * delta;    
    
    // variables used for calculating event rates
            vec_double      global_r_rates(n_reactions);        // global rate of each reaction
            double          propensity_sum;                     // sum of global reaction rates


    // init a random generator
    std::mt19937 gen(std::random_device{}());


    double t = time_points[0];
    // loop through the time points to sample
    for (int i = 0; i < n_time_points; ++i) {
        
        while (t < time_points[i]) {
                
            // avoiding negative values, calculate dynamic birth rates in nodes with active birthrate control, and set corresponding reaction rates
            for (int j = 0; j < n_birthrate_updates; j+=2) {
                percap_r_rates[j] = percap_r_rates[j+1] = std::max(0.0, (mu + c_b*(nss - sys_state[j] - (delta*sys_state[j+1])))); 
            }
            
            // calculate global reaction propensity by multiplyin per capita rates with the number of reactants, while keeping track of their cumsum
            propensity_sum = 0.0;
            for (int j = 0; j < n_reactions; j++) {
                global_r_rates[j] = percap_r_rates[j]*sys_state[state_index[j]];
                propensity_sum += global_r_rates[j];
            }

            // if the system as died out, break the loop
            if (propensity_sum == 0.0) {
                t = time_points[i];
                break;
            }

            // get the reaction and apply the reaction to the state of the system
            std::discrete_distribution<> react_pmf(global_r_rates.begin(), global_r_rates.end());
            sys_state += reactions.row(react_pmf(gen));          
            
            // increment time forward
            std::exponential_distribution<> expdist(propensity_sum);
            t += expdist(gen);
 
                
        }
        
        // write the current state of the system to the output array
        sys_state_sample.row(i) = sys_state;
        
    }
}


// TAU LEAPING FUNCTION //
void sim_tauleaping(
    const   np_vec_f64      in_time_points,             // points in time where system state should be recorded
    const   double          timestep,                   // simulation time step
    
    const   np_vec_i32      in_sys_state,               // starting 'sys_state'
            np_mat_i32      sys_state_sample,           // array where the 'sys_state' is recorded at each time in 'time_points'
            
    const   np_mat_i32      reactions,                  // 'sys_state' updates corresponding to each possible reaction
    const   np_vec_f64      in_percap_r_rates,             // per capita reaction rates
    const   np_vec_i32      in_state_index,             // indeces of system state variables from which propensity is calculated

    const   np_vec_f64      in_birthrate_updates_par,   // parameters used to update dynamic birth rates
    const   int             n_birthrate_updates         // number of birth rate reactions which must be updated
    )
{   
    // ### VARIABLE SETUP #### //

    // c++ versions of numpy arrays
    const   vec_double      time_points     = get_double_vec_from_np(in_time_points);
            vec_double      percap_r_rates  = get_double_vec_from_np(in_percap_r_rates);
    const   vec_int         state_index     = get_int_vec_from_np(in_state_index);
            Eigen::VectorXi sys_state       = get_eig_int_vec_from_np(in_sys_state);
            
    // counts often accessed during loops
    const   int             n_time_points   = time_points.size();
    const   int             n_reactions     = state_index.size();
    const   int             n_pops          = sys_state.size();

    // variables used in calculating the dynamic birth rates
    const   vec_double      br_up_par       = get_double_vec_from_np(in_birthrate_updates_par);
    const   double          c_b             = br_up_par[0];
    const   double          mu              = br_up_par[1];
    const   double          nss             = br_up_par[2];
    const   double          delta           = br_up_par[3];
    const   double          brh1            = mu + c_b * nss;   // precomputed values that do not change across iterations
    const   double          brh2            = c_b * delta;
    
    // variables used to derive the rate of poisson processes
            Eigen_diag_mat  diagonalized_n_events(n_reactions); // a diagonalized matrix where each element represents the number of times a given reaction occures
            Eigen::MatrixXi product(n_reactions, n_pops);       // a matrix matrix product of the diagonalized event count matrix and the reaction matrix


    // ### SIMULATOR #### //

    // init a random generator
    std::mt19937 gen(std::random_device{}());

    
    double t = time_points[0]; 
    // loop through the time points to sample
    for (int i = 0; i < n_time_points; ++i) {

        // check if the system has completely exhausted, and exit if needed (input array is all 0s anyway)
        if (sys_state.isZero()) {
            break;
        }

        // while the next time point to sample the system state is reached
        while (t < time_points[i]) {
            
            // avoiding negative values, calculate dynamic birth rates in nodes with active birthrate control, and set corresponding reaction rates
            for (int j = 0; j < n_birthrate_updates; j+=2) {
                percap_r_rates[j] = percap_r_rates[j+1] = std::max(0.0, (mu + c_b*(nss - sys_state[j] - (delta*sys_state[j+1]))));  
            }

            // calculate rates, and use as mean of poisson, and then generate the number of times each reaction occurs during the timestep.
            for (int j = 0; j < n_reactions; ++j) {
                std::poisson_distribution<> dist(percap_r_rates[j]*sys_state[state_index[j]]*timestep);
                diagonalized_n_events.diagonal()(j) = dist(gen);
            }

            // update the state of the system by adding each reaction the correct number of times                     
            product = diagonalized_n_events*reactions;              
            sys_state += product.colwise().sum();         
            sys_state = sys_state.cwiseMax(0); // guarantee that there are no underflows
            
            // increment time forward
            t += timestep;
        }
        
        // write the current state of the system to the output array
        sys_state_sample.row(i) = sys_state;

    }
}

// PYBIND // 
PYBIND11_MODULE(libsdesim, m)
{
    m.def("sim_gillespie", &sim_gillespie, "simulate using gillespie");
    m.def("sim_tauleaping", &sim_tauleaping, "simulate using tau leaping");
}