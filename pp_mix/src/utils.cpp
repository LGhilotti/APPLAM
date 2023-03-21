#include "utils.hpp"

void delete_row(MatrixXd *x, int ind) {
  int nrow = x->rows() - 1;
  int dim = x->cols();
  if (ind < nrow) {
    x->block(ind, 0, nrow - ind, dim) = x->block(ind + 1, 0, nrow - ind, dim);
  }

  x->conservativeResize(nrow, dim);
}

void delete_column(MatrixXd *x, int ind) {
  int nrow = x->rows();
  int dim = x->cols()-1;
  if (ind < dim) {
    x->block(0, ind, nrow, dim-ind) = x->block(0,ind+1, nrow , dim-ind);
  }

  x->conservativeResize(nrow, dim);
}

void delete_elem(VectorXd *x, int ind) {
  int size = x->size() - 1;
  if (ind < size) x->segment(ind, size - ind) = x->segment(ind + 1, size - ind);

  x->conservativeResize(size);
}

MatrixXd delete_row(const MatrixXd &x, int ind) {
  MatrixXd out = x;
  int nrow = x.rows() - 1;
  int dim = x.cols();
  if (ind < nrow) {
    out.block(ind, 0, nrow - ind, dim) = out.block(ind + 1, 0, nrow - ind, dim);
  }

  out.conservativeResize(nrow, dim);
  return out;
}

MatrixXd delete_column(const MatrixXd x, int ind) {
  MatrixXd out = x;
  int nrow = x.rows();
  int dim = x.cols()-1;
  if (ind < dim) {
    out.block(0, ind, nrow, dim-ind) = out.block(0,ind+1, nrow , dim-ind);
  }

  out.conservativeResize(nrow, dim);
  return out;
}

VectorXd delete_elem(const VectorXd &x, int ind) {
  VectorXd out = x;
  int size = x.size() - 1;
  if (ind < size)
    out.segment(ind, size - ind) = out.segment(ind + 1, size - ind);

  out.conservativeResize(size);
  return out;
}

MatrixXd vstack(const std::vector<VectorXd> &rows) {
  int nrows = rows.size();
  int ncols = rows[0].size();

  MatrixXd out(nrows, ncols);
  for (int i = 0; i < nrows; i++) out.row(i) = rows[i].transpose();

  return out;
}


double o_multi_normal_prec_lpdf(const VectorXd &x, const VectorXd &mu,
                                const PrecMat &sigma) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;

  double out = 0.5 * sigma.get_log_det() + NEG_LOG_SQRT_TWO_PI * x.size();
  out -= 0.5 * (sigma.get_cho_factor_eval() * (x - mu)).squaredNorm();
  return out;
}

double o_multi_normal_prec_lpdf(const std::vector<VectorXd> &x,
                                const VectorXd &mu, const PrecMat &sigma) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;

  int n = x.size();
  double out = sigma.get_log_det() * n;

  const MatrixXd &cho_sigma = sigma.get_cho_factor_eval();

  std::vector<double> loglikes(n);
  for (int i = 0; i < n; i++) {
    loglikes[i] = (cho_sigma * (x[i] - mu)).squaredNorm();
  }

  out -= std::accumulate(loglikes.begin(), loglikes.end(), 0.0);

  return 0.5 * out;
}

void to_proto(const MatrixXd &mat, EigenMatrix *out) {
  out->set_rows(mat.rows());
  out->set_cols(mat.cols());
  *out->mutable_data() = {mat.data(), mat.data() + mat.size()};
}

void to_proto(const VectorXd &vec, EigenVector *out) {
  out->set_size(vec.size());
  *out->mutable_data() = {vec.data(), vec.data() + vec.size()};
}

VectorXd to_eigen(const EigenVector &vec) {
  int size = vec.size();
  Eigen::VectorXd out;
  if (size > 0) {
    const double *p = &(vec.data())[0];
    out = Map<const VectorXd>(p, size);
  }
  return out;
}

MatrixXd to_eigen(const EigenMatrix &mat) {
  int nrow = mat.rows();
  int ncol = mat.cols();
  Eigen::MatrixXd out;
  if (nrow > 0 & ncol > 0) {
    const double *p = &(mat.data())[0];
    out = Map<const MatrixXd>(p, nrow, ncol);
  }
  return out;
}


std::vector<VectorXd> to_vector_of_vectors(const MatrixXd &mat) {
  std::vector<VectorXd> out(mat.rows());
  for (int i = 0; i < mat.rows(); i++) out[i] = mat.row(i).transpose();

  return out;
}

VectorXd softmax_fun(const VectorXd &logs) {

  VectorXd num = VectorXd::Zero(logs.size());
  for (int j = 0; j<logs.size(); j++){
    if (!(isinf(logs(j)) || isnan(logs(j)))){
      num(j)=exp((logs(j)-logs.maxCoeff()));
    }
  }
  //std::cout<<"num: "<<num<<std::endl;
  if (num.sum() == 0)
    return num;
  else  return num / num.sum();
  // after a "strange" step, what it returns is (p1/P,...,pM/P), where P=sum(pi)
  // what it receives is (ln(p1),...,ln(pM))
}


Eigen::MatrixXd posterior_similarity(
    const Eigen::MatrixXd &alloc_chain) {
  unsigned int n_data = alloc_chain.cols();
  Eigen::MatrixXd mean_diss = Eigen::MatrixXd::Zero(n_data, n_data);
  // Loop over pairs (i,j) of data points
  for (int i = 0; i < n_data; i++) {
    for (int j = 0; j < i; j++) {
      Eigen::ArrayXd diff = alloc_chain.col(i) - alloc_chain.col(j);
      mean_diss(i, j) = (diff == 0).count();
    }
  }
  return mean_diss / alloc_chain.rows();
}
