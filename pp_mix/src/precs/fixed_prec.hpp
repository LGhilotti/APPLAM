#ifndef FIXED_COV_HPP
#define FIXED_COV_HPP

#include "base_prec.hpp"

class Delta_FixedUniv : public BaseUnivPrec
{
protected:
   double sigma;

public:
   Delta_FixedUniv(double sigma) : sigma(sigma) {}

   ~Delta_FixedUniv() {}

   double sample_prior() override { return sigma; }

   double sample_alloc(
       const std::vector<double> &data, const double &curr,
       const VectorXd &mean) override { return sigma; }

   double mean() const override { return sigma; }

   double lpdf(double val) const override {return 0.0; };
};

class Delta_FixedMulti : public BaseMultiPrec
{
protected:
   int dim;
   double sigma; // The precision matrix is assumed to be sigma*I here

public:
   Delta_FixedMulti(int dim, double sigma) : dim(dim), sigma(sigma) {}

   ~Delta_FixedMulti() {}

   PrecMat sample_prior() override;

   PrecMat sample_alloc(
       const std::vector<VectorXd> &data, const PrecMat &curr,
       const VectorXd &mean) override;

   PrecMat mean() const override;

   double lpdf(const PrecMat &val) const override {return 0.0; };
};


#endif
