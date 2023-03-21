#ifndef LAMBDA_SAMPLER_HPP
#define LAMBDA_SAMPLER_HPP

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
}*/
#include "conditional_mcmc.hpp"

namespace MCMCsampler {

class BaseLambdaSampler {
protected:
    MultivariateConditionalMCMC* mcmc;

    int acc_sampled_Lambda = 0;
    int tot_sampled_Lambda = 0;

    MatrixXd grad_log_ad;
    MatrixXd grad_log_analytic;
    double ln_dens_ad;
    double ln_dens_analytic;

public:
    BaseLambdaSampler(MultivariateConditionalMCMC* mcmc): mcmc(mcmc) {}
    virtual ~BaseLambdaSampler(){}

    virtual void perform(MatrixXd& Ctilde) = 0;

    const MatrixXd& get_grad_log_ad(){return grad_log_ad;}
    const MatrixXd& get_grad_log_analytic(){return grad_log_analytic;}
    double get_ln_dens_ad(){return ln_dens_ad;}
    double get_ln_dens_analytic(){return ln_dens_analytic;}

    double Lambda_acc_rate();
};


class LambdaSamplerClassic : public BaseLambdaSampler {
private:
    double prop_lambda_sigma;

     // for update of Lambda!
    inline double compute_exp_lik(const MatrixXd& lamb) const;
    inline double compute_exp_prior(const MatrixXd& lamb) const;

public:
    LambdaSamplerClassic(MultivariateConditionalMCMC* mcmc, double p_l_s): BaseLambdaSampler(mcmc), prop_lambda_sigma(p_l_s){}
    void perform(MatrixXd& Ctilde) override;
};

class LambdaSamplerMala : public BaseLambdaSampler {
private:
    double mala_p_lambda;

    double compute_ln_dens_analytic(const MatrixXd& lamb, double);
    double compute_ln_dens_analytic(double);
    MatrixXd compute_grad_analytic(const MatrixXd& lamb, const MatrixXd& Ctilde);
    MatrixXd compute_grad_analytic(const MatrixXd& Ctilde);
    MatrixXd compute_gr_an(const MatrixXd& lamb, const MatrixXd& Ctilde, const VectorXd& Phis, double Ds);
public:
    LambdaSamplerMala(MultivariateConditionalMCMC* mcmc, double m_p): BaseLambdaSampler(mcmc), mala_p_lambda(m_p), lambda_tar_fun(*mcmc){}
    void perform(MatrixXd& Ctilde) override;

    // TARGET FUNCTION OBJECT : must implement logfunction (as required in Mala)
    class lambda_target_function {
    private:
        const MultivariateConditionalMCMC& m_mcmc;

    public:

        lambda_target_function(const MultivariateConditionalMCMC& mala): m_mcmc(mala){};

        template<typename T> T
        operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & lamb) const ;
    } lambda_tar_fun;
};

}

#include "lambda_sampler_imp.hpp"

#endif
