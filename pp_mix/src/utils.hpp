#ifndef UTILS_HPP
#define UTILS_HPP

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/delimited_message_util.h>

#include <Eigen/Dense>
#include <fstream>
#include <numeric>
#include <random>
#include <stan/math/prim.hpp>
#include <vector>

#include "../protos/cpp/state.pb.h"
#include "precs/precmat.hpp"

using namespace Eigen;
using std::vector;

void delete_row(MatrixXd *x, int ind);

void delete_column(MatrixXd *x, int ind);

void delete_elem(VectorXd *x, int ind);

MatrixXd delete_row(const MatrixXd &x, int ind);

MatrixXd delete_column(const MatrixXd &x, int ind);

VectorXd delete_elem(const VectorXd &x, int ind);

MatrixXd vstack(const std::vector<VectorXd> &rows);

// used for testing the code in spikes folder
template <typename T>
T loadTextProto(std::string filename) {
  std::ifstream ifs(filename);
  google::protobuf::io::IstreamInputStream iis(&ifs);
  T out;
  auto success = google::protobuf::TextFormat::Parse(&iis, &out);
  if (!success)
    std::cout << "An error occurred in 'loadTextProto'; success: " << success
              << std::endl;
  return out;
}

double o_multi_normal_prec_lpdf(const VectorXd &x, const VectorXd &mu,
                                const PrecMat &sigma);


// this is just proportional (-n*p/2 log(2pi) misses): we just need it for MH step so it is enough
double o_multi_normal_prec_lpdf(const std::vector<VectorXd> &x,
                                const VectorXd &mu, const PrecMat &sigma);

void to_proto(const MatrixXd &mat, EigenMatrix *out);

void to_proto(const VectorXd &vec, EigenVector *out);

Eigen::VectorXd to_eigen(const EigenVector &vec);

Eigen::MatrixXd to_eigen(const EigenMatrix &vec);

std::vector<VectorXd> to_vector_of_vectors(const MatrixXd &mat);

VectorXd softmax_fun(const VectorXd &logs);

Eigen::MatrixXd posterior_similarity(const Eigen::MatrixXd &alloc_chain);

template<typename T>
vector<vector<T>> cart_product(const vector<vector<T>> &v) {
  vector<vector<T>> s = {{}};
  for (const auto &u : v) {
    vector<vector<T>> r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = move(r);
  }
  return vector<vector<T>>(s.end()-(s.size()/2 +1), s.end());
}

template <typename T>
vector<vector<T>> cart_product(const vector<T> &v, int times) {
  vector<vector<T>> vecs(times);
  for (int i=0; i < times; i++)
    vecs[i] = v;

  return cart_product(vecs);
}

#endif
