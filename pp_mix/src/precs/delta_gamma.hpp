#ifndef DELTA_GAMMA_HPP
#define DELTA_GAMMA_HPP

#include <algorithm>
#include <stan/math/prim.hpp>

#include "base_prec.hpp"
#include "../rng.hpp"
#include "../utils.hpp"

class Delta_Gamma : public BaseUnivPrec {
 protected:
  double alpha;
  double beta;

 public:
  Delta_Gamma(double alpha, double beta);

  ~Delta_Gamma() {}

  double sample_prior() override;

  double sample_alloc(const std::vector<double> &data, const double &curr,
                           const VectorXd &mean) override;

  double mean() const override;

  double lpdf(double val) const override {
    return stan::math::gamma_lpdf(val, alpha, beta);
  };
};

#endif
