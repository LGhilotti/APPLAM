#include "fixed_prec.hpp"


PrecMat Delta_FixedMulti::sample_prior()
{
    Eigen::MatrixXd out = sigma * Eigen::MatrixXd::Identity(dim, dim);
    return PrecMat(out);
}

PrecMat Delta_FixedMulti::sample_alloc(
    const std::vector<VectorXd> &data, const PrecMat &curr,
    const VectorXd &mean)
{
    return sample_prior();
}

PrecMat Delta_FixedMulti::mean() const
{
    Eigen::MatrixXd out = sigma * Eigen::MatrixXd::Identity(dim, dim);
    return PrecMat(out);
}
