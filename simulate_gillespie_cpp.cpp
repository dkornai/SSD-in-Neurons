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

namespace py = pybind11;


// set up shorthands for types
using int64 = long;
using float64 = double;

using int64_1d = std::vector<int64>;
using float64_1d = std::vector<float64>;

typedef Eigen::Matrix<long, -1, -1> MatrixXl;

float64_1d get_flt_vec_from_np(
    const   py::array_t<double> arr
    )
{
    py::buffer_info info = arr.request();
    double* data = static_cast<double*>(info.ptr);
    std::size_t size = info.size;
    std::vector<double> vec(data, data + size);
    return vec;
}

int64_1d get_int_vec_from_np(
    const   py::array_t<int64>  arr
    )
{
    py::buffer_info info = arr.request();
    long* data = static_cast<long*>(info.ptr);
    std::size_t size = info.size;
    std::vector<long> vec(data, data + size);
    return vec;
}


// draw samples from an exponential with a specified mean 
float64 random_sample_exponential(
    const   float64     mean
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
    const   float64_1d& react_probal
    ) 
{
    // initialized only once the first time the function is called
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // sample from the p.m.f. defined by 'react_probal'
    std::discrete_distribution<> dist(react_probal.begin(), react_probal.end());
    return dist(gen);
}


// MAIN GILLESPIE FUNCTION
void simulate(
    const   py::array_t<double> v_time_points,
            py::array_t<double> v_react_rates,
    const   py::array_t<long>   v_state_index,
            py::array_t<long>   v_sys_state, 
    const   int64       n_pops,
    const   int64       n_reactions,
            Eigen::Ref<MatrixXl> sys_state_sample,
    const   Eigen::Ref<MatrixXl> reactions,
    const   py::array_t<double> v_birthrate_updates_par
    )
{
    
    const   float64_1d  time_points = get_flt_vec_from_np(v_time_points);
            float64_1d  react_rates = get_flt_vec_from_np(v_react_rates);
    const   int64_1d    state_index = get_int_vec_from_np(v_state_index);
            int64_1d    sys_state   = get_int_vec_from_np(v_sys_state);
    const   float64_1d  birthrate_updates_par = get_flt_vec_from_np(v_birthrate_updates_par);

    const int n_time_points = time_points.size();

    const float64     c_b = birthrate_updates_par[0];
    const float64     mu = birthrate_updates_par[1];
    const float64     nss = birthrate_updates_par[2];
    const float64     delta = birthrate_updates_par[3];
    
    float64     birth_rate;
    int64       reaction_index;

    float64_1d  react_propen(n_reactions);
    float64_1d  react_probal(n_reactions);

    float64     propensity_sum = 0.0;
    float64     t = time_points[0];

    // test section
    if (sys_state[0] != 9) {
        throw std::runtime_error("Test failed!");
    }


    for (int i = 0; i < n_time_points; ++i) {
        // while (t < time_points[i]) {
        //     // update birth rates in compartment 0 (main node)
        //     birth_rate = mu + c_b*(nss-sys_state[0]-(delta*sys_state[1]));
        //     if (birth_rate < 0) // check for negative rates (possible if e.p.s. > nss)
        //         birth_rate = 0; 
        //     react_rates[0] = react_rates[1] = birth_rate;

        //     propensity_sum = 0.0;
        //     for (int j = 0; j < n_reactions; j++) {
        //         // calculate global reaction propensity by multiplyin per capita rates with the number of reactants
        //         react_propen[j] = react_rates[j]*sys_state[state_index[j]];
        //         // keep track of the sum of global propensities
        //         propensity_sum += react_propen[j];
        //     }

        //     // if there exist any reactions (e.g. all reaction rates > 0, all reactants present at > 0)
        //     if (propensity_sum != 0.0) {
        //         // normalize global propensities such that their sum is 1 (forming a pmf)
        //         for (int j = 0; j < n_reactions; j++) {
        //             react_probal[j] = react_propen[j]/propensity_sum;
        //         }
                
        //         // randomly generate a reaction
        //         reaction_index = random_event_pmf(react_probal);  
            
        //         // apply the reaction to the state of the system
        //         for (int j = 0; j < n_pops; j++) {
        //             sys_state[j] += reactions(reaction_index,j);          
        //         }
                
        //         // increment time forward
        //         t += random_sample_exponential(1/propensity_sum);
            
        //     // if no reactions occur with probability > 0 (the system is empty), the state of the system will never change
        //     } else {
        //         t += 0.001;
        //     }
        // }
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