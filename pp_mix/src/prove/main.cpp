//#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
//#include <stan/math/prim.hpp>
#include <stan/math/fwd.hpp>
#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>

#include "../../conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../precs/delta_gamma.hpp"
#include "../../precs/delta_wishart.hpp"
#include "../../point_process/determinantalPP.hpp"

using namespace Eigen;
using namespace stan::math;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

int main{

  MatrixXd datacsv = load_csv<MatrixXd>("../../../../data/Student_latent_data/applam/dataset.csv");

  std::string params_file = \
      "../../../../data/Student_latent_data/resources/sampler_params_aniso.asciipb";
  Params params = loadTextProto<Params>(params_file);

  int d = 2;
  matrixXd ranges(2,2);
  ranges << -6 , -6, 6, 6;

  BaseDeterminantalPP* pp_mix = make_dpp(params, ranges);

  BasePrec* g = make_delta(params, d);

  MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params, d);

  std::vector<int> init_allocs(datacsv.rows());
  std::iota (std::begin(init_allocs), std::end(init_allocs), 0);

  Eigen::VectorXi init_allocs_ = Eigen::Map<Eigen::VectorXi>(init_allocs.data(), init_allocs.size());
  MatrixX lamb = MatrixXd::Zero(datacsv.cols(), d);

  sampler.initialize(datacsv, init_allocs_, lamb);

  int niter= 10;
  int thin = 1;
  int log_every=1;

  for (int i = 0; i < niter; i++) {
    sampler.run_one(fix_lambda = "TRUE", fix_sigma = "TRUE");
    if ((i + 1) % log_every == 0) {
      std::cout<<"Running, iter #", i + 1, " / ", niter << std::endl;
      std::cout<< sampler.get_clus_alloc()<< std::endl;

      /*Eigen::VectorXi ca = sampler.get_clus_alloc();
      for (int h=0; h< ca.size(); h++){
        py::print(ca(h));
      }*/
    }
  }

  std::cout<<"acceptance all means: "<<sampler.a_means_acceptance_rate()<<std::endl;



  return 0;
}
