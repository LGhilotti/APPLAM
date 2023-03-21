#include <deque>
#include <string>
#include <tuple>
//#include <stan/math/fwd.hpp>
//#include <stan/math/mix.hpp>
//#include <stan/math/prim.hpp>
//#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
/// COLPA DI QUESTO EIGEN.H
//#include <pybind11/eigen.h>


#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"
#include "conditional_mcmc.hpp"
#include "factory.hpp"
#include "utils.hpp"


namespace py = pybind11;

/////////////////////////// This is the main function provided to Python user

std::tuple<std::deque<py::bytes>, double , double>
 _run_pp_mix(int ntrick, int burnin, int niter, int thin,
                                  std::string serialized_data, //const Eigen::MatrixXd &data,
                                  std::string serialized_params,
                                  int d,
                                  std::string serialized_ranges, //const Eigen::MatrixXd &ranges,
                                  std::vector<int> init_allocs,
				  int log_every = 200) {
  Params params;
  params.ParseFromString(serialized_params);
  Eigen::MatrixXd data;
  Eigen::MatrixXd ranges;

  {
    EigenMatrix data_proto;
    data_proto.ParseFromString(serialized_data);
    data = to_eigen(data_proto);
    EigenMatrix ranges_proto;
    ranges_proto.ParseFromString(serialized_ranges);
    ranges = to_eigen(ranges_proto);
  }

  std::deque<py::bytes> out;
  DeterminantalPP* pp_mix = make_dpp(params, ranges);
  BasePrec* g = make_delta(params, d);

  MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params, d);
  sampler.initialize(data);

  Eigen::VectorXi init_allocs_ = Eigen::Map<Eigen::VectorXi>(init_allocs.data(), init_allocs.size());
  sampler.set_clus_alloc(init_allocs_);
  sampler._relabel();
  py::print("Number means in trick phase: ", sampler.get_num_a_means());

  for (int i = 0; i < ntrick; i++) {
    sampler.run_one_trick();
    if ((i + 1) % log_every == 0) {
      py::print("Trick, iter #", i + 1, " / ", ntrick);
    }
  }

  for (int i = 0; i < burnin; i++) {
    sampler.run_one();
    if ((i + 1) % log_every == 0) {
      py::print("Burnin, iter #", i + 1, " / ", burnin);
    }
  }

  for (int i = 0; i < niter; i++) {
    sampler.run_one();
    if (i % thin == 0) {
      std::string s;
      MultivariateMixtureState curr;
      sampler.get_state_as_proto(&curr);
      curr.SerializeToString(&s);
      out.push_back((py::bytes)s);
    }

    if ((i + 1) % log_every == 0) {
      py::print("Running, iter #", i + 1, " / ", niter);
    }
  }

  py::object Decimal = py::module_::import("decimal").attr("Decimal");

  py::print("Allocated Means acceptance rate ", Decimal(sampler.a_means_acceptance_rate()));
  py::print("Lambda acceptance rate ", Decimal(sampler.Lambda_acceptance_rate()));

  return std::make_tuple(out,sampler.a_means_acceptance_rate(),sampler.Lambda_acceptance_rate());
}

/////////////////////////// This is the main function provided to Python user FOR 0/1 RESPONSES

std::tuple<std::deque<py::bytes>, double , double>
 _run_pp_mix_binary(int ntrick, int burnin, int niter, int thin,
                                  std::string serialized_data, //const Eigen::MatrixXi &data,
                                  std::string serialized_params,
                                  int d,
                                  std::string serialized_ranges, //const Eigen::MatrixXd &ranges,
                                  std::vector<int> init_allocs,
				  int log_every = 200) {
  Params params;
  params.ParseFromString(serialized_params);
  Eigen::MatrixXd data;
  Eigen::MatrixXd ranges;

  {
    EigenMatrix data_proto;
    data_proto.ParseFromString(serialized_data);
    data = to_eigen(data_proto);
    EigenMatrix ranges_proto;
    ranges_proto.ParseFromString(serialized_ranges);
    ranges = to_eigen(ranges_proto);
  }

  std::deque<py::bytes> out;
  DeterminantalPP* pp_mix = make_dpp(params, ranges);
  BasePrec* g = make_delta(params, d);

  MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params, d);
  sampler.initialize_binary(data);

  Eigen::VectorXi init_allocs_ = Eigen::Map<Eigen::VectorXi>(init_allocs.data(), init_allocs.size());
  sampler.set_clus_alloc(init_allocs_);
  sampler._relabel();
  py::print("Number means in trick phase: ", sampler.get_num_a_means());
  py::print("initial mean abs zetas: ", sampler.get_data().array().abs().mean());

  for (int i = 0; i < ntrick; i++) {
    //py::print("current mean abs zetas: ", sampler.get_data().array().abs().mean());
    //py::print("current mean abs etas: ", sampler.get_etas().array().abs().mean());

    sampler.run_one_trick_binary();
    if ((i + 1) % log_every == 0) {
      py::print("Trick, iter #", i + 1, " / ", ntrick);
    }
  }

  for (int i = 0; i < burnin; i++) {
    //py::print("current mean abs zetas: ", sampler.get_data().array().abs().mean());
    //py::print("current mean abs etas: ", sampler.get_etas().array().abs().mean());

    sampler.run_one_binary();
    if ((i + 1) % log_every == 0) {
      py::print("Burnin, iter #", i + 1, " / ", burnin);
    }
  }

  for (int i = 0; i < niter; i++) {
    sampler.run_one_binary();
    if (i % thin == 0) {
      std::string s;
      MultivariateMixtureState curr;
      sampler.get_state_as_proto(&curr);
      curr.SerializeToString(&s);
      out.push_back((py::bytes)s);
    }

    if ((i + 1) % log_every == 0) {
      py::print("Running, iter #", i + 1, " / ", niter);
    }
  }

  py::object Decimal = py::module_::import("decimal").attr("Decimal");

  py::print("Allocated Means acceptance rate ", Decimal(sampler.a_means_acceptance_rate()));
  py::print("Lambda acceptance rate ", Decimal(sampler.Lambda_acceptance_rate()));

  return std::make_tuple(out,sampler.a_means_acceptance_rate(),sampler.Lambda_acceptance_rate());
}



//params: coll Collector containing the algorithm chain
//return: Index of the iteration containing the best estimate
py::bytes cluster_estimate( //const Eigen::MatrixXd &alloc_chain
            std::string serialized_alloc_chain) {
  // Initialize objects
  Eigen::MatrixXd alloc_chain;
  {
    EigenMatrix alloc_chain_proto;
    alloc_chain_proto.ParseFromString(serialized_alloc_chain);
    alloc_chain = to_eigen(alloc_chain_proto);
  }
  unsigned n_iter = alloc_chain.rows();
  unsigned int n_data = alloc_chain.cols();
  std::vector<Eigen::SparseMatrix<double> > all_diss;
  //progresscpp::ProgressBar bar(n_iter, 60);

  // Compute mean
  std::cout << "(Computing mean dissimilarity... " << std::flush;
  Eigen::MatrixXd mean_diss = posterior_similarity(alloc_chain);
  std::cout << "Done)" << std::endl;

  // Compute Frobenius norm error of all iterations
  Eigen::VectorXd errors(n_iter);
  for (int k = 0; k < n_iter; k++) {
    for (int i = 0; i < n_data; i++) {
      for (int j = 0; j < i; j++) {
        int x = (alloc_chain(k, i) == alloc_chain(k, j));
        errors(k) += (x - mean_diss(i, j)) * (x - mean_diss(i, j));
      }
    }
    // Progress bar
    //++bar;
    //bar.display();
  }
  //bar.done();

  // Find iteration with the least error
  std::ptrdiff_t ibest;
  unsigned int min_err = errors.minCoeff(&ibest);
  Eigen::VectorXd out = alloc_chain.row(ibest).transpose();
  std::string out_string;
  EigenVector out_proto;
  to_proto(out, &out_proto);
  out_proto.SerializeToString(&out_string);

  return (py::bytes)out_string;

}



PYBIND11_MODULE(pp_mix_high, m) {
  m.doc() = "aaa";  // optional module docstring

  m.def("_run_pp_mix", &_run_pp_mix, "aaa");

  m.def("_run_pp_mix_binary", &_run_pp_mix_binary, "aaa");

  m.def("cluster_estimate", &cluster_estimate, "aaa");
}
