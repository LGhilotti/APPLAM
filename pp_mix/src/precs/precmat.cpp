#include "precmat.hpp"


PrecMat::PrecMat(const MatrixXd &prec): prec(prec) {
    cho_factor = LLT<MatrixXd>(prec);
    cho_factor_eval = cho_factor.matrixL().transpose();
    log_det = 2 * log(cho_factor_eval.determinant());
    if (compute_var)
        var = cho_factor.solve(MatrixXd::Identity(prec.rows(), prec.cols()));

    return;
}

MatrixXd PrecMat::get_prec() const
{
    return prec;
}

MatrixXd PrecMat::get_var() const {
    if (! compute_var)
        throw std::runtime_error("Variance has not been computed!");

    return var;
}

LLT<MatrixXd> PrecMat::get_cho_factor() const
{
    return cho_factor;
}

MatrixXd PrecMat::get_cho_factor_eval() const
{
    return cho_factor_eval;
}

double PrecMat::get_log_det() const
{
    return log_det;
}

std::ostream &operator << (std::ostream & output, const PrecMat &p) {
    const Eigen::MatrixXd mat = p.get_prec();
    return output << mat << "\n";
}
