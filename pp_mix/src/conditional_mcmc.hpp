#ifndef MULTI_CONDITIONAL_MCMC
#define MULTI_CONDITIONAL_MCMC

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <functional>
#include <cmath>


//include <stan/math/fwd.hpp>
//#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>

#include <google/protobuf/message.h>

#include "rng.hpp"
#include "point_process/determinantalPP.hpp"
#include "precs/base_prec.hpp"
#include "utils.hpp"
#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"


using namespace Eigen;
using namespace stan::math;

namespace MCMCsampler {
    class BaseLambdaSampler ;
    class BaseMeansSampler ;
}


namespace MCMCsampler {

class MultivariateConditionalMCMC {
 protected:
    int dim_fact;
    int dim_data;
    int ndata;
    MatrixXd data; //also called data if latent, when binary responses: data are there called binary_data
    MatrixXi binary_data;
    // fixed hyperparameters
    double _a_phi; // Dirichlet parameter
    double _alpha_jump, _beta_jump; // Gamma jump parameters
    double _a_gamma, _b_gamma; // Sigma_bar parameters

    // for each allocated cluster, it contains the vector of indexes of observations:
    // both useful for data and etas.
    std::vector<std::vector<int>> obs_by_clus;

    // NOTE: in each iteration (run_one), when updating etas, this structure is no more correct,
    // since it is not updated. But this structure is no more used in following updates, so
    // it is ok. This will be updated at next iteration in _relabel and used correctly
    std::vector<std::vector<VectorXd>> etas_by_clus;

    /* STATE */
    // Rep-pp
    double u;
    VectorXi clus_alloc;
    VectorXd a_jumps, na_jumps;
    MatrixXd a_means, na_means;
    std::vector<PrecMat> a_deltas, na_deltas;
    // etas: n x d matrix
    MatrixXd etas;

    // Sigma_bar
    VectorXd sigma_bar;
    //Lambda-block
    double tau;
    MatrixXd Phi;
    MatrixXd Psi;
    MatrixXd Lambda;

    //C_tilde
    MatrixXd Ctilde;

    // Lambda sampling callable object
    BaseLambdaSampler* sample_lambda;
    // Allocated means sampling callable object
    BaseMeansSampler* sample_means_obj;


    // FOR DEBUGGING
    bool verbose = false;

    Params params;


 public:

    // DISTRIBUTIONS
    DeterminantalPP *pp_mix;
    BaseMultiPrec *g;


    MultivariateConditionalMCMC() {}
    ~MultivariateConditionalMCMC() ;

    MultivariateConditionalMCMC(DeterminantalPP *pp_mix, BasePrec *g,
                                const Params &params, int d);


    void set_pp_mix(DeterminantalPP* pp_mix) {this->pp_mix = pp_mix;}
    void set_prec(BaseMultiPrec* g) {this->g = g;}
    void set_params(const Params & p, int d);

    // initializes some of the members (data, dim, ndata,..) and state of sampler
    // The constructor only initialize some other members (pointers) and params field
    void initialize(const MatrixXd& dat);
    void initialize_binary(const MatrixXd& binary_dat);

    // initializes the etas, projecting the data onto Col(Lambda):
    // it is for both uni/multi factor cases, but implemented differently because of the least square systems.
    void initialize_etas(const MatrixXd &dat);
    void initialize_latent_data(const MatrixXi& binary_dat);

    void initialize_allocated_means();

    std::vector<VectorXd> proj_inside();
    bool is_inside(const VectorXd & point);

    // it performs the whole step of updatings
    void run_one();
    void run_one_trick();

    void run_one_binary();
    void run_one_trick_binary();
    void sample_latent_data();
    MatrixXd trunc_normal_rng(const ArrayXXd& means, const ArrayXXd& sigmas,
                              const ArrayXXi& y);

    // SAMPLING (UPDATE) METHODS
    // REP-PP BLOCK
    // sample non-allocated jumps
    void sample_jumps_na();
    // sample allocated jumps
    void sample_jumps_a();
    // sample non-allocated means
    void sample_means_na(double psi_u);
    //sample non-allocated means with trick, without changing number
    void sample_means_na_trick();
    // sample allocated means: OBJECT FOR MANAGE MALA AND MH
    //void sample_means_a();
    // sample non-allocated deltas
    void sample_deltas_na();
    // sample allocated deltas
    void sample_deltas_a();
    // sample cluster allocations and relabel the parameters
    void sample_allocations_and_relabel();

    void _relabel();

    // sample u
    inline void sample_u(){
      double T = a_jumps.sum() + na_jumps.sum();
      u = gamma_rng(ndata, T, Rng::Instance().get());
    };

    // ETAS: virtual because for dim_fact=1, we directly invert scalars, not resolving systems!
    void sample_etas();

    // SIGMA_BAR: could be virtual because for dim_fact=1, we can exploit that eta^T eta is scalar..
    void sample_sigma_bar();

    // LAMBDA BLOCK : identical for both uni/multi factor since uni uses a matrixXd with 1 column.
    void sample_Psi();
    void sample_tau();
    void sample_Phi();
    // virtual: in uni cond process does not depend on Lambda: OBJECT TO MANAGE MALA AND MH
    //virtual void sample_Lambda() = 0;

    // will be private
    inline double laplace(double u) const {
        return std::pow(_beta_jump, _alpha_jump) / std::pow(_beta_jump + u, _alpha_jump);
    }


    // to store the current state in proto format
    void get_state_as_proto(google::protobuf::Message *out_);

    // to just print the current state for debugging
    void print_debug_string();

    void set_verbose() { verbose = !verbose; }

    double lpdf_given_clus(const VectorXd &x, const VectorXd &mu, const PrecMat &sigma)
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }

    double lpdf_given_clus_multi(
        const std::vector<VectorXd> &x, const VectorXd &mu, const PrecMat &sigma)
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }

    // Returns a matrix with entries
    //   out[i, j] = k(y_j, tau_i) = N(y_j, Lambda mu_i, Lambda Delta_i Lambda^T)
    MatrixXd data_lpdf_in_components();

    //virtual VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) = 0;

    double a_means_acceptance_rate();

    double Lambda_acceptance_rate();

    void print_data_by_clus(int clus);

    // getters
    int get_ndata() const {return ndata;}

    int get_dim_data() const {return dim_data;}

    int get_dim_fact() const {return dim_fact;}

    double get_u() const {return u;}

    double get_tau() const {return tau;}

    const MatrixXd& get_Psi() const {return Psi;}

    const MatrixXd& get_Phi() const {return Phi;}

    const MatrixXd& get_Lambda() const {return Lambda;}

    const VectorXd& get_sigma_bar() const {return sigma_bar;}

    const MatrixXd& get_data() const {return data;}

    const MatrixXd& get_etas() const {return etas;}

    const std::vector<VectorXd>& get_etas_by_clus(int ind) const {return etas_by_clus[ind];}

    int get_num_a_means() const {return a_means.rows();}

    int get_num_na_means() const {return na_means.rows();}

    RowVectorXd get_single_a_mean(int ind) const {return a_means.row(ind);}

    RowVectorXd get_single_na_mean(int ind) const {return na_means.row(ind);}

    const PrecMat& get_single_a_delta(int ind) const {return a_deltas[ind];}

    const PrecMat& get_single_na_delta(int ind) const {return na_deltas[ind];}

    MatrixXd get_a_means_except_ind(int ind) const {return delete_row(a_means, ind);}

    const MatrixXd& get_a_means() const {return a_means;}

    const MatrixXd& get_na_means() const {return na_means;}

    MatrixXd get_all_means() const {
        MatrixXd out(a_means.rows()+na_means.rows(), dim_fact);
        out << a_means , na_means;
        return out;
    }

    MatrixXd get_all_means_trans() const {
        return pp_mix->get_A()*get_all_means().transpose() + pp_mix->get_b();
    }

    MatrixXd get_all_means_reverse() const {
        MatrixXd out(na_means.rows()+a_means.rows(), dim_fact);
        out << na_means , a_means;
        return out;
    }

    VectorXi get_clus_alloc() const { return clus_alloc;  }

    VectorXd get_a_jumps() const { return a_jumps;  }

    VectorXd get_na_jumps() const { return na_jumps;  }

    void set_Lambda(const MatrixXd& prop_lambda) {  Lambda = prop_lambda;}

    void set_single_a_mean(int ind, const VectorXd& prop) { a_means.row(ind) = prop.transpose() ;}

    void set_single_na_mean(int ind, const VectorXd& prop) { na_means.row(ind) = prop.transpose() ;}

    double get_ln_dens_ad() ;

    double get_ln_dens_analytic() ;

    const MatrixXd& get_grad_log_ad();

    const MatrixXd& get_grad_log_analytic();

    MatrixXd get_ctilde() const {return Ctilde;}

    void set_clus_alloc(const Eigen::VectorXi _allocs) {
	    clus_alloc = _allocs;
    }

};



}


#endif
