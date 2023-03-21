#ifndef FACTORY_HPP
#define FACTORY_HPP


#include "lambda_sampler.hpp"
#include "alloc_means_sampler.hpp"

#include "precs/base_prec.hpp"
#include "precs/fixed_prec.hpp"
#include "precs/delta_wishart.hpp"
#include "precs/delta_gamma.hpp"

#include "point_process/determinantalPP.hpp"

#include "../protos/cpp/params.pb.h"

// Lambda sampler
MCMCsampler::BaseLambdaSampler* make_LambdaSampler(MCMCsampler::MultivariateConditionalMCMC* mcmc, const Params& params);

// AMeans sampler
MCMCsampler::BaseMeansSampler* make_MeansSampler(MCMCsampler::MultivariateConditionalMCMC* mcmc, const Params& params);

// DPP
DeterminantalPP* make_dpp(const Params& params, const MatrixXd& ranges);
DeterminantalPP* make_dpp(const Params& params, int d);

// Delta Precision
BasePrec* make_delta(const Params& params, int d);

BasePrec *make_fixed_prec(const FixedMultiPrecParams &params, int d);

BasePrec* make_wishart(const WishartParams& params, int d);

BasePrec* make_fixed_prec(const FixedUnivPrecParams& params);

BasePrec* make_gamma_prec(const GammaParams& params);

#endif
