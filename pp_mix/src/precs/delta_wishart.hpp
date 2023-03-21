#ifndef DELTA_WISHART_HPP
#define DELTA_WISHART_HPP

#include <vector>
#include <Eigen/Dense>
#include <stan/math/prim.hpp>

#include "base_prec.hpp"
#include "../rng.hpp"
#include "../utils.hpp"

using namespace Eigen;
using namespace stan::math;

class Delta_Wishart : public BaseMultiPrec {
protected:
  double df; // called n on wiki
  MatrixXd psi; // called V on wiki
  MatrixXd inv_psi;

public:
  Delta_Wishart(double df, int dim, double sigma); // assumes psi= sigma*I

  ~Delta_Wishart(){};

  // Sample from full-cond of Delta^(na): non allocated are distributed as prior
  PrecMat sample_prior() override;

  // Sample from full-cond of Delta^(a): allocated
  PrecMat sample_alloc(
      const std::vector<VectorXd> &data, const PrecMat &curr,
      const VectorXd &mean) override;

  PrecMat mean() const override;

  double get_df() const {return df;}

  MatrixXd get_psi() const {return psi;}

  double lpdf(const PrecMat& val) const override {
      return stan::math::wishart_lpdf(val.get_prec(), df, psi);
  };

};

#endif
