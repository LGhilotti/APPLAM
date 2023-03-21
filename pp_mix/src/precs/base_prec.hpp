#ifndef BASE_PREC_HPP
#define BASE_PREC_HPP

#include <vector>
#include <Eigen/Dense>

#include "precmat.hpp"

using namespace Eigen;

class BasePrec
{
public:
    virtual ~BasePrec(){};
};

class BaseUnivPrec: public BasePrec {
public:
    virtual ~BaseUnivPrec() {};

    // Sample from full-cond of Delta^(na): non allocated are distributed as prior
    virtual double sample_prior() = 0;

    // Sample from full-cond of Delta^(a): allocated
    virtual double sample_alloc(
        const std::vector<double> &data, const double &curr,
        const VectorXd &mean) = 0;

    virtual double mean() const = 0; // prior mean

    virtual double lpdf(double val) const = 0;
};


class BaseMultiPrec: public BasePrec {
public:
    virtual ~BaseMultiPrec(){};

    // Sample from full-cond of Delta^(na): non allocated are distributed as prior
    virtual PrecMat sample_prior() = 0;

    // Sample from full-cond of Delta^(a): allocated
    virtual PrecMat sample_alloc(
        const std::vector<VectorXd> &data, const PrecMat &curr,
        const VectorXd &mean) = 0;

    virtual PrecMat mean() const = 0;

    virtual double lpdf(const PrecMat &val) const = 0;
};

#endif
