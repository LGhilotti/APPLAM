#ifndef PREC_MAT_HPP
#define PREC_MAT_HPP

#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;


class PrecMat {
 protected:
   MatrixXd prec;
   MatrixXd var;
   LLT<MatrixXd> cho_factor;
   MatrixXd cho_factor_eval; // matrix U in Cholesky decomposition: Prec=LL^T = U^T U
   double log_det;
   double univariate_val;
   bool is_univariate = false;
   bool compute_var = false;

 public:
   PrecMat() {}
   ~PrecMat() {}

   PrecMat(const MatrixXd &prec);

   PrecMat(const double &prec): univariate_val(prec) {
      is_univariate = true;
   }

   void set_compute_var() {
     compute_var = true;
   }

   MatrixXd get_prec() const;

   MatrixXd get_var() const;

   LLT<MatrixXd> get_cho_factor() const;

   MatrixXd get_cho_factor_eval() const;

   double get_log_det() const;
};

std::ostream &operator<<(std::ostream &output, const PrecMat &p);

#endif
