#ifndef ALLOC_MEANS_SAMPLER
#define ALLOC_MEANS_SAMPLER

#include <omp.h>
#include <stan/math/fwd.hpp>
#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>
#include "rng.hpp"

using namespace Eigen;
using namespace stan::math;
/*
namespace MCMCsampler {
    class MultivariateConditionalMCMC ;
}
*/
#include "conditional_mcmc.hpp"

namespace MCMCsampler {

class BaseMeansSampler {
protected:
    MultivariateConditionalMCMC* mcmc;

    // only for allocated
    int acc_sampled_a_means = 0;
    int tot_sampled_a_means = 0;


public:
    BaseMeansSampler(MultivariateConditionalMCMC* mcmc): mcmc(mcmc) {}
    virtual ~BaseMeansSampler(){}

    virtual void perform_update_allocated(MatrixXd& ) = 0;

    virtual void perform_update_trick_na(MatrixXd& ) = 0;


    double Means_acc_rate(); // only accounts for allocated
};


class MeansSamplerClassic : public BaseMeansSampler {
private:
    double prop_means_sigma;

public:
    MeansSamplerClassic(MultivariateConditionalMCMC* mcmc, double p_m_s): BaseMeansSampler(mcmc), prop_means_sigma(p_m_s){}

    void perform_update_allocated(MatrixXd& ) override;

    void perform_update_trick_na(MatrixXd& ) override;

};

class MeansSamplerMala : public BaseMeansSampler {
private:
    double mala_p_means;
    MatrixXd allmeans;
    int ind_mean;

public:

    MeansSamplerMala(MultivariateConditionalMCMC* mcmc, double m_p): BaseMeansSampler(mcmc), mala_p_means(m_p),
    alloc_means_tar_fun(*this), trick_na_means_tar_fun(*this){}

    void perform_update_allocated(MatrixXd& ) override;

    void perform_update_trick_na(MatrixXd& ) override;

    const MatrixXd& get_allmeans() const {return allmeans;}
    int get_num_allmeans() const {return allmeans.rows();}
    int get_ind_mean() const {return ind_mean;}

    // ALLOC MEANS TARGET FUNCTION OBJECT : must implement logfunction (as required in Mala)
    class alloc_means_target_function {
    private:
        //const MultivariateConditionalMCMC& m_mcmc;
        //const MatrixXd& all_means;
        //const int& ind_mean;
        const MeansSamplerMala& msm;
    public:

        alloc_means_target_function(const MeansSamplerMala& _msm): msm(_msm){};

        template<typename T> T
        operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & mean) const ;
    } alloc_means_tar_fun;


    // TRICK NA MEANS TARGET FUNCTION OBJECT : must implement logfunction (as required in Mala)
    class trick_na_means_target_function {
    private:
        //const MultivariateConditionalMCMC& m_mcmc;
        //const MatrixXd& all_means;
        //const int& ind_mean;
        const MeansSamplerMala& msm;
    public:

        trick_na_means_target_function(const MeansSamplerMala& _msm): msm(_msm){};

        template<typename T> T
        operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & mean) const ;
    } trick_na_means_tar_fun;
};

};

#include "alloc_means_sampler_imp.hpp"

#endif
