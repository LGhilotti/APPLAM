#include "conditional_mcmc.hpp"
#include "lambda_sampler.hpp"
#include "alloc_means_sampler.hpp"
#include "factory.hpp"
#include "gig.hpp"
#include "rng.hpp"

namespace MCMCsampler {

MultivariateConditionalMCMC::MultivariateConditionalMCMC(DeterminantalPP *pp_mix,
                                                         BasePrec *g,
                                                         const Params &params,
                                                         int d)
    : pp_mix(pp_mix) {

  set_prec(dynamic_cast<BaseMultiPrec *>(g));
  set_params(params,d);
  sample_lambda = make_LambdaSampler(this, params);
  sample_means_obj = make_MeansSampler(this, params);

}

MultivariateConditionalMCMC::~MultivariateConditionalMCMC()
{
         delete pp_mix;
         delete g;
         delete sample_lambda;
         delete sample_means_obj;
}

void MultivariateConditionalMCMC::set_params(const Params & p, int d){

  this->params = p;
  this->dim_fact = d;
  this->_a_phi = params.a();
  this->_alpha_jump = params.alphajump();
  this->_beta_jump = params.betajump();
  this->_a_gamma = params.agamma();
  this->_b_gamma = params.bgamma();

  return;
}


void MultivariateConditionalMCMC::initialize(const MatrixXd& dat) {

  this->data = dat;
  ndata = data.rows();
  dim_data = data.cols();

  // Initialize Lambda block: tau, psi, phi and Lambda
  tau = 2.0 * dim_data * dim_fact * _a_phi ;
  Phi = 1.0/(dim_data*dim_fact) * MatrixXi::Ones(dim_data,dim_fact) ;
  Psi = 2.0 * MatrixXi::Ones(dim_data,dim_fact);

  Lambda = Map<MatrixXd>(normal_rng( std::vector<double>(dim_data*dim_fact, 0.0),
        std::vector<double>(dim_data*dim_fact, _a_phi * _a_phi), Rng::Instance().get() ).data() , dim_data,dim_fact );

  // Initialize Sigma_bar
  sigma_bar = _a_gamma/_b_gamma * VectorXd::Ones(dim_data);

  // Initialize etas
  initialize_etas(dat);

  // REP-PP BLOCK
  // Initialize the allocated means
  initialize_allocated_means();
  //std::cout << "a_means: \n" << a_means.transpose() << std::endl;

  double nclus = a_means.rows();
  // Initialize cluster allocations
  clus_alloc.resize(ndata);
  // initial vector s(a) has identical element and sums to nclus/(nclus+1)
  a_jumps = VectorXd::Ones(nclus) / (nclus); // nclus+1 if consider 1 non allocated comp

  // initial vector Delta(a) is equal to the mean (scalar or matrix)
  a_deltas.resize(nclus);
  for (int i = 0; i < nclus; i++) {
    a_deltas[i] = g->mean();
    // std::cout << "prec: \n" << a_deltas[i] << std::endl;
  }

  // initial mu(na) is just one, uniformly selected in ranges
  //na_means = pp_mix->sample_uniform(1);
  na_means.resize(0,dim_fact);
  // initial s(na) (just one value) is 1/(nclus + 1) -> this way the full s vector sums to 1!
  //na_jumps = VectorXd::Ones(na_means.rows()) / (nclus + na_means.rows());
  na_jumps.resize(0);

  // initial Delta(na) (just one scalar or matrix) is the mean
  na_deltas.resize(na_means.rows());
  /*for (int i = 0; i < na_means.rows(); i++) {
    na_deltas[i] = g->mean();
  }*/
  // initial u parameter
  u = 1.0;

  // DECOMPOSE DPP (in MultiDpp also assign the pointer to Lambda)
  pp_mix->set_decomposition(&Lambda);

  // TEST UPDATE RELABEL
  Ctilde = pp_mix->compute_Ctilde(get_all_means());


}


void MultivariateConditionalMCMC::initialize_binary(const MatrixXd& binary_dat) {

  ndata = binary_dat.rows();
  dim_data = binary_dat.cols();

  this->binary_data = binary_dat;

  // Initialize Lambda block: tau, psi, phi and Lambda
  tau = 2.0 * dim_data * dim_fact * _a_phi ;
  Phi = 1.0/(dim_data*dim_fact) * MatrixXi::Ones(dim_data,dim_fact) ;
  Psi = 2.0 * MatrixXi::Ones(dim_data,dim_fact);

  Lambda = Map<MatrixXd>(normal_rng( std::vector<double>(dim_data*dim_fact, 0.0),
        std::vector<double>(dim_data*dim_fact, _a_phi * _a_phi), Rng::Instance().get() ).data() , dim_data,dim_fact );

  // Initialize Sigma_bar
  sigma_bar = _a_gamma/_b_gamma * VectorXd::Ones(dim_data);

  // initialize latent data
  initialize_latent_data(binary_dat);

  // Initialize etas
  initialize_etas(data);

  // REP-PP BLOCK
  // Initialize the allocated means
  initialize_allocated_means();
  //std::cout << "a_means: \n" << a_means.transpose() << std::endl;

  double nclus = a_means.rows();
  // Initialize cluster allocations
  clus_alloc.resize(ndata);
  // initial vector s(a) has identical element and sums to nclus/(nclus+1)
  a_jumps = VectorXd::Ones(nclus) / (nclus); // nclus+1 if consider 1 non allocated comp

  // initial vector Delta(a) is equal to the mean (scalar or matrix)
  a_deltas.resize(nclus);
  for (int i = 0; i < nclus; i++) {
    a_deltas[i] = g->mean();
    // std::cout << "prec: \n" << a_deltas[i] << std::endl;
  }

  // initial mu(na) is just one, uniformly selected in ranges
  //na_means = pp_mix->sample_uniform(1);
  na_means.resize(0,dim_fact);
  // initial s(na) (just one value) is 1/(nclus + 1) -> this way the full s vector sums to 1!
  //na_jumps = VectorXd::Ones(na_means.rows()) / (nclus + na_means.rows());
  na_jumps.resize(0);

  // initial Delta(na) (just one scalar or matrix) is the mean
  na_deltas.resize(na_means.rows());
  /*for (int i = 0; i < na_means.rows(); i++) {
    na_deltas[i] = g->mean();
  }*/
  // initial u parameter
  u = 1.0;

  // DECOMPOSE DPP (in MultiDpp also assign the pointer to Lambda)
  pp_mix->set_decomposition(&Lambda);

  // TEST UPDATE RELABEL
  Ctilde = pp_mix->compute_Ctilde(get_all_means());


}

void MultivariateConditionalMCMC::initialize_etas(const MatrixXd &dat) {

  LLT<MatrixXd> M (Lambda.transpose() * Lambda);

  etas = (M.solve((dat*Lambda).transpose())).transpose();

  return;

}

void MultivariateConditionalMCMC::initialize_latent_data(const MatrixXi& binary_dat){
  MatrixXd m = (binary_dat.array() - 0.5)*2;
  std::vector<double> means( m.data(), m.data() + m.size() );

  data = Map<MatrixXd>(normal_rng( means,
        std::vector<double>(ndata*dim_data, 0.01), Rng::Instance().get() ).data() , ndata,dim_data );
}


void MultivariateConditionalMCMC::initialize_allocated_means() {
  int init_n_clus = 8;
  std::vector<VectorXd> in = proj_inside();

  if (init_n_clus >= in.size()) {
    a_means.resize(in.size(), dim_fact);
    for (int i=0; i < in.size(); i++)
      a_means.row(i) = in[i].transpose();
  } else {
    a_means.resize(init_n_clus, dim_fact);
    std::vector<int> index(in.size());
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.begin() + in.size(),std::default_random_engine(1234) );
    for (int i=0; i < init_n_clus; i++) {
      a_means.row(i) = in[index[i]].transpose();
    }
  }
  //std::cout<<"alloc means: "<<a_means.rows()<<std::endl;
  return;
}

std::vector<VectorXd> MultivariateConditionalMCMC::proj_inside() {

  std::vector<VectorXd> inside;
  for (int i=0; i<ndata;i++){
    if (is_inside(etas.row(i))){
      inside.push_back(etas.row(i));
    }
  }
  return inside;
}

bool MultivariateConditionalMCMC::is_inside(const VectorXd & point){
  MatrixXd ran = pp_mix->get_ranges();
  bool is_in = true;
  for (int i=0;i<dim_fact&& is_in==true;i++){
    if (point(i)<ran(0,i) || point(i)>ran(1,i)){
      is_in = false;
    }
  }
  return is_in;
}

void MultivariateConditionalMCMC::run_one() {

  //std::cout<<"sample u"<<std::endl;
  sample_u();

  //std::cout<<"compute psi"<<std::endl;

  // compute laplace transform psi in u
  double psi_u = laplace(u);

  //std::cout<<"sample alloca and relabel"<<std::endl;
  // sample c | rest and reorganize the all and nall parameters, and c as well
  sample_allocations_and_relabel();

  //std::cout<<"sample means na"<<std::endl;
  // sample non-allocated variables
  // I can set Ctilde with all the means and then remove or add row/column when na_means is updated.
  //Ctilde = pp_mix->compute_Ctilde(get_all_means());
  sample_means_na(psi_u);

  //std::cout<<"sample jumps na"<<std::endl;
  sample_jumps_na();

  //std::cout<<"sample deltsa na"<<std::endl;
  sample_deltas_na();

  //std::cout<<"sample means a"<<std::endl;
  // sample allocated variables
  sample_means_obj->perform_update_allocated(Ctilde);

  //std::cout<<"sample deltas a"<<std::endl;
  sample_deltas_a();

  //std::cout<<"sample jumps a"<<std::endl;
  sample_jumps_a();

  //// TEST
  //std::cout<<"means_trans:\n"<<  ((pp_mix->get_A() * get_all_means().transpose()).colwise() + pp_mix->get_b()).transpose()<<std::endl;
  //std::cout<<"Ctilde_after_alloc:\n"<<Ctilde<<std::endl;

  // sample etas
  //std::cout << "sample etas" << std::endl;
  sample_etas();

  // sample Sigma bar
  //std::cout << "sample etas" << std::endl;
  sample_sigma_bar();

  // sample Lambda block
  sample_Psi();
  sample_tau();
  sample_Phi();

  sample_lambda->perform(Ctilde);

  // print_debug_string();

  return;
}


void MultivariateConditionalMCMC::run_one_trick() {

//  std::cout<<"sample u"<<std::endl;
  sample_u();

  //std::cout<<"compute psi"<<std::endl;

  //std::cout<<"sample alloca and relabel"<<std::endl;
  // sample c | rest and reorganize the all and nall parameters, and c as well
  // sample_allocations_and_relabel();

  //std::cout<<"sample means na"<<std::endl;
  sample_means_obj->perform_update_trick_na(Ctilde);

  //std::cout<<"sample jumps na"<<std::endl;
  sample_jumps_na();

  //std::cout<<"sample deltsa na"<<std::endl;
  sample_deltas_na();

//  std::cout<<"sample means a"<<std::endl;
  sample_means_obj->perform_update_allocated(Ctilde);

//  std::cout<<"sample deltas a"<<std::endl;
  sample_deltas_a();

//  std::cout<<"sample jumps a"<<std::endl;
  sample_jumps_a();

//  std::cout<<"sample etas"<<std::endl;
  sample_etas();

//  std::cout<<"sample sigmabar"<<std::endl;
  sample_sigma_bar();

//  std::cout<<"sample Psi"<<std::endl;
  sample_Psi();

//  std::cout<<"sample tau"<<std::endl;
  sample_tau();

//  std::cout<<"sample Phi"<<std::endl;
  sample_Phi();

//  std::cout<<"before sampling Lambda"<<std::endl;
  sample_lambda->perform(Ctilde);
 //std::cout<<"sample Lambda"<<std::endl;

  // print_debug_string();

  return;
}

/////////////////////////////////////////////////////////////////
//////////// FOR BINARY SAMPLER //////////////////////////////////
////////////////////////////////////////////////////////////////

void MultivariateConditionalMCMC::run_one_binary() {

  sample_latent_data();

  //std::cout<<"sample u"<<std::endl;
  sample_u();

  //std::cout<<"compute psi"<<std::endl;

  // compute laplace transform psi in u
  double psi_u = laplace(u);

  //std::cout<<"sample alloca and relabel"<<std::endl;
  // sample c | rest and reorganize the all and nall parameters, and c as well
  sample_allocations_and_relabel();

  //std::cout<<"sample means na"<<std::endl;
  // sample non-allocated variables
  // I can set Ctilde with all the means and then remove or add row/column when na_means is updated.
  //Ctilde = pp_mix->compute_Ctilde(get_all_means());
  sample_means_na(psi_u);

  //std::cout<<"sample jumps na"<<std::endl;
  sample_jumps_na();

  //std::cout<<"sample deltsa na"<<std::endl;
  sample_deltas_na();

  //std::cout<<"sample means a"<<std::endl;
  // sample allocated variables
  sample_means_obj->perform_update_allocated(Ctilde);

  //std::cout<<"sample deltas a"<<std::endl;
  sample_deltas_a();

  //std::cout<<"sample jumps a"<<std::endl;
  sample_jumps_a();

  //// TEST
  //std::cout<<"means_trans:\n"<<  ((pp_mix->get_A() * get_all_means().transpose()).colwise() + pp_mix->get_b()).transpose()<<std::endl;
  //std::cout<<"Ctilde_after_alloc:\n"<<Ctilde<<std::endl;

  // sample etas
  //std::cout << "sample etas" << std::endl;
  sample_etas();

  // sample Sigma bar
  //std::cout << "sample etas" << std::endl;
  // FIXED TO THE IDENTITY FOR THE APPLICATION
  //sample_sigma_bar();

  // sample Lambda block
  sample_Psi();
  sample_tau();
  sample_Phi();

  sample_lambda->perform(Ctilde);

  // print_debug_string();

  return;
}

void MultivariateConditionalMCMC::run_one_trick_binary() {
  //sample_latent_data();

  sample_u();

  //std::cout<<"compute psi"<<std::endl;

  //std::cout<<"sample alloca and relabel"<<std::endl;
  // sample c | rest and reorganize the all and nall parameters, and c as well
  // sample_allocations_and_relabel();

  sample_means_obj->perform_update_trick_na(Ctilde);

  //std::cout<<"sample jumps na"<<std::endl;
  sample_jumps_na();

  //std::cout<<"sample deltsa na"<<std::endl;
  sample_deltas_na();

  //std::cout<<"sample means a"<<std::endl;
  sample_means_obj->perform_update_allocated(Ctilde);

  //std::cout<<"sample deltas a"<<std::endl;
  sample_deltas_a();

  //std::cout<<"sample jumps a"<<std::endl;
  sample_jumps_a();

  //std::cout<<"sample etas"<<std::endl;
  sample_etas();

  //std::cout<<"sample sigmabar"<<std::endl;
  // FIXED TO THE IDENTITY FOR THE APPLICATION
  //sample_sigma_bar();

  //std::cout<<"sample Psi"<<std::endl;
  sample_Psi();

  //std::cout<<"sample tau"<<std::endl;
  sample_tau();

  //std::cout<<"sample Phi"<<std::endl;
  sample_Phi();

  //std::cout<<"before sampling Lambda"<<std::endl;
  sample_lambda->perform(Ctilde);
 //std::cout<<"sample Lambda"<<std::endl;

  // print_debug_string();

  return;
}


void MultivariateConditionalMCMC::sample_latent_data(){


  ArrayXXd sigmas = (1/sigma_bar.array()).sqrt().matrix().transpose().replicate(ndata,1);

  data = trunc_normal_rng((Lambda*etas.transpose()).transpose(), sigmas, binary_data);
}

MatrixXd MultivariateConditionalMCMC::trunc_normal_rng(const ArrayXXd& means, const ArrayXXd& sigmas,
                          const ArrayXXi& y){ // y are the data y_i in {0,1}

  int n_data = means.rows();
  int dim_data = means.cols();
  int n_times_p = means.size();

  /* inverse CDF method */

  // sample U from uniform(0,1)

  ArrayXXd u = Map<ArrayXXd>( uniform_rng( std::vector<double>(n_times_p, 0),
              std::vector<double>(n_times_p, 1), Rng::Instance().get()).data(), n_data, dim_data);

  ArrayXXd t1 = stan::math::Phi(-means/sigmas);

  return sigmas*stan::math::inv_Phi(t1*u + (1-u)*y) + means;

}

//////////////////////////////////////////////////////////////////////////////////////

void MultivariateConditionalMCMC::sample_jumps_na()
{

    na_jumps.conservativeResize(na_means.rows(), 1);
    int N_na = na_jumps.size();
    na_jumps = Map<VectorXd>( gamma_rng( std::vector<double>(N_na, _alpha_jump),
            std::vector<double>(N_na, _beta_jump + u) , Rng::Instance().get()).data() , N_na);
    return;
}


void MultivariateConditionalMCMC::sample_jumps_a()
{

    //#pragma omp parallel for
    for (int i = 0; i < a_means.rows(); i++)
      a_jumps(i) = gamma_rng(_alpha_jump+etas_by_clus[i].size(), _beta_jump+u , Rng::Instance().get());

    return;
}

void MultivariateConditionalMCMC::sample_means_na(double psi_u)
{

  for (int i=0; i < 10; i++) {
    int na_points = na_means.rows();
    pp_mix->sample_nonalloc_fullcond(&na_means, a_means, psi_u, Ctilde);
    //std::cout<<"rep "<<i<<": "<<na_means<<std::endl;
    if (na_means.rows() != na_points)
      break;

  }//at this point also Ctilde is updated to current updated means.
  return;
}

/*
void MultivariateConditionalMCMC::sample_means_na_trick()
{
  MatrixXd allmeans(na_means.rows() + a_means.rows(), dim_fact);
  allmeans << na_means, a_means;

  for (int i = 0; i < na_means.rows(); i++) {
    //tot_sampled_a_means += 1;
    MatrixXd others(allmeans.rows() - 1, dim_fact);

    double prior_ratio;
    const VectorXd &currmean = na_means.row(i).transpose();
    const MatrixXd &cov_prop = MatrixXd::Identity(dim_fact, dim_fact) * prop_means_sigma;

    // we PROPOSE a new point from a multivariate normal, with mean equal to the current point
    // and covariance matrix diagonal
    VectorXd prop =
        stan::math::multi_normal_rng(currmean, cov_prop, Rng::Instance().get());

    if (is_inside(prop)){ // if not, just keep the current mean and go to the next a_mean
        others = delete_row(allmeans, i);

        prior_ratio =
            pp_mix->papangelou(prop, others) - pp_mix->papangelou(currmean, others);

        if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < prior_ratio) {
          na_means.row(i) = prop.transpose();
          //acc_sampled_a_means += 1;
        }
    }
  }
  return;
}
*/

void MultivariateConditionalMCMC::sample_deltas_na() {

  na_deltas.resize(na_means.rows());

  for (int i = 0; i < na_means.rows(); i++) {
    na_deltas[i] = g->sample_prior();
  }
  return;
}


void MultivariateConditionalMCMC::sample_deltas_a() {

  //#pragma omp parallel for
  for (int i = 0; i < a_means.rows(); i++) {
    a_deltas[i] = g->sample_alloc(etas_by_clus[i], a_deltas[i],
                                      a_means.row(i).transpose());
  }
  return;
}



void MultivariateConditionalMCMC::sample_allocations_and_relabel() {
  int Ma = a_means.rows();
  int Mna = na_means.rows();
  // current number of components (a + na)
  int Mtot = Ma + Mna;

  // std::cout << "a_means: \n" << a_means << std::endl;

  const MatrixXd &curr_a_means = a_means;
  const MatrixXd &curr_na_means = na_means;
  const std::vector<PrecMat> &curr_a_deltas = a_deltas;
  const std::vector<PrecMat> &curr_na_deltas = na_deltas;
  const VectorXd &curr_a_jumps = a_jumps;
  const VectorXd &curr_na_jumps = na_jumps;

  std::vector<MatrixXd> a_prec_matrices;
  std::vector<MatrixXd> na_prec_matrices;

  MatrixXd log_probas = data_lpdf_in_components();
  for (int k = 0; k < Ma; k++) {
      log_probas.row(k) = log_probas.row(k).array() + std::log(a_jumps(k));
  }

  for (int k = 0; k < Mna; k++) {
      log_probas.row(k + Ma) = log_probas.row(k + Ma).array() + std::log(na_jumps(k));
  }

  for (int i = 0; i < ndata; i++) {
    VectorXd probas = softmax_fun(log_probas.col(i));
    if (probas.sum() == 0)
      clus_alloc[i] = categorical_rng(VectorXd::Constant(Ma + Mna,1), Rng::Instance().get()) - 1;
    else clus_alloc[i] = categorical_rng(probas, Rng::Instance().get()) - 1;
  }

  _relabel();
}

MatrixXd MultivariateConditionalMCMC::data_lpdf_in_components() {

    MatrixXd M0(Lambda.transpose() * sigma_bar.asDiagonal());
    MatrixXd M1( M0 * Lambda);

    MatrixXd out(a_jumps.size() + na_jumps.size(), data.rows());
    int d = a_means.cols();

    for (int i = 0; i < a_jumps.size(); i++) {
      VectorXd mu = Lambda * a_means.row(i).transpose();
      MatrixXd Sn_bar_cho = LLT<MatrixXd>(
        M1 + a_deltas[i].get_prec()).solve(MatrixXd::Identity(d, d));
      MatrixXd M2 = M0 * (data.rowwise() - mu.transpose()).transpose();
      VectorXd num = (Sn_bar_cho * M2).colwise().squaredNorm();
      MatrixXd temp = (data.rowwise() - mu.transpose()).array().square();
      num += temp * sigma_bar;;
      num = -0.5 * num;
      double log_det_prec = sigma_bar.array().log().sum() + \
        a_deltas[i].get_log_det() + \
        Sn_bar_cho.diagonal().array().log().sum();

      out.row(i) = num.array() + 0.5 * log_det_prec;
    }

    for (int i = 0; i < na_jumps.size(); i++) {
      VectorXd mu = Lambda * na_means.row(i).transpose();
      MatrixXd Sn_bar_cho = LLT<MatrixXd>(
        M1 + a_deltas[i].get_prec()).solve(MatrixXd::Identity(d, d));
      MatrixXd M2 = M0 * (data.rowwise() - mu.transpose()).transpose();
      VectorXd num = (Sn_bar_cho * M2).colwise().squaredNorm();
      MatrixXd temp = (data.rowwise() - mu.transpose()).array().square();
      num += temp * sigma_bar;
      num = -0.5 * num;

      double log_det_prec = sigma_bar.array().log().sum() + \
        na_deltas[i].get_log_det() + \
        Sn_bar_cho.diagonal().array().log().sum();

      out.row(a_jumps.size() + i) = num.array() + 0.5 * log_det_prec;
    }

    return out;
}

void MultivariateConditionalMCMC::_relabel() {
  std::set<int> na2a;  // non active that become active
  std::set<int> a2na;  // active that become non active

  int Ma = a_means.rows();
  int Mna = na_means.rows();
  int Mtot = Ma + Mna;

  // FILL THE NA2A
  for (int i = 0; i < ndata; i++) {
    if (clus_alloc(i) >= Ma) na2a.insert(clus_alloc(i) - Ma);
  }

  // FILL THE A2NA
  for (int k = 0; k < Ma; k++) {
    if ((clus_alloc.array() == k).count() == 0) a2na.insert(k);
  }

  std::vector<int> a2na_vec(a2na.begin(), a2na.end());
  int n_new_na = a2na.size();
  MatrixXd new_na_means(n_new_na, dim_fact);
  std::vector<PrecMat> new_na_deltas(n_new_na);
  VectorXd new_na_jumps(n_new_na);

  for (int i = 0; i < n_new_na; i++) {
    new_na_means.row(i) = a_means.row(a2na_vec[i]);
    new_na_deltas[i] = a_deltas[a2na_vec[i]];
    new_na_jumps(i) = a_jumps(a2na_vec[i]);
  }

  // NOW TAKE CARE OF NON ACTIVE THAT BECOME ACTIVE
  std::vector<int> na2a_vec(na2a.begin(), na2a.end());
  int n_new_a = na2a_vec.size();
  MatrixXd new_a_means(n_new_a, dim_fact);
  std::vector<PrecMat> new_a_deltas(n_new_a);
  VectorXd new_a_jumps(n_new_a);

  for (int i = 0; i < n_new_a; i++) {
    new_a_means.row(i) = na_means.row(na2a_vec[i]);
    new_a_deltas[i] = na_deltas[na2a_vec[i]];
    double tmp = na_jumps(na2a_vec[i]);
    new_a_jumps(i) = tmp;
  }

  // delete rows, backward
  for (auto it = a2na_vec.rbegin(); it != a2na_vec.rend(); it++) {
    delete_row(&a_means, *it);
    delete_elem(&a_jumps, *it);
    a_deltas.erase(a_deltas.begin() + *it);
  }

  // delete rows, backward
  if (na_means.rows() > 0) {
    for (auto it = na2a_vec.rbegin(); it != na2a_vec.rend(); it++) {
      delete_row(&na_means, *it);
      delete_elem(&na_jumps, *it);
      na_deltas.erase(na_deltas.begin() + *it);
    }
  }

  // NOW JOIN THE STUFF TOGETHER
  if (new_a_means.rows() > 0) {
    int oldMa = a_means.rows();
    a_means.conservativeResize(oldMa + new_a_means.rows(), dim_fact);
    a_means.block(oldMa, 0, new_a_means.rows(), dim_fact) = new_a_means;

    a_jumps.conservativeResize(oldMa + new_a_means.rows());
    a_jumps.segment(oldMa, new_a_means.rows()) = new_a_jumps;

    for (const auto &prec : new_a_deltas) a_deltas.push_back(prec);
  }

  if (new_na_means.rows() > 0) {
    int oldMna = na_means.rows();
    na_means.conservativeResize(oldMna + new_na_means.rows(), dim_fact);
    na_means.block(oldMna, 0, new_na_means.rows(), dim_fact) = new_na_means;

    na_jumps.conservativeResize(oldMna + new_na_means.rows());
    na_jumps.segment(oldMna, new_na_means.rows()) = new_na_jumps;
  }

  for (const auto &prec : new_na_deltas) na_deltas.push_back(prec);

  // BUILD THE VECTOR OF PERMUTATION

  std::vector<int> perm(Mtot);
  for (int j=0; j< Ma; j++){
    int num=0;
    if (n_new_na==0){
      perm[j]=j;
    } else for (int i=0; i< n_new_na; i++){
      if (a2na_vec[i]==j){
        perm[j]=Mtot - n_new_na + i;
        break;
      } else if (a2na_vec[i] < j & num < n_new_na) {
        num++;
      }
      if (a2na_vec[i]>j || num == n_new_na) {
        perm[j] = j - num;
        break;
      }
    }

  }

  for (int j=Ma; j<Mtot; j++){
    bool found=false;
    int count=0;
    for (int i=0; i< n_new_a; i++){
      if (na2a_vec[i]+Ma==j){
        perm[j]=Ma - n_new_na + i;
        found=true;
        break;
      } else if (na2a_vec[i]+Ma<j){
        count++;
      } else break;
    }
    if (found==false){
      perm[j] = j + n_new_a - count - n_new_na;
    }
  }

  //now the perm vector tells the new positions of the means.
  MatrixXd Ctilde_up = MatrixXd::Zero(Mtot,Mtot);
  for (int i=0; i<Mtot-1; i++){
    for (int j=i+1; j<Mtot; j++){
      Ctilde_up(perm[i],perm[j])=Ctilde(i,j);
    }
  }
  MatrixXd Ctilde_up_tmp = Ctilde_up + Ctilde_up.transpose();
  Ctilde_up_tmp.diagonal() = Ctilde.diagonal();

  Ctilde = Ctilde_up_tmp;

  // NOW RELABEL
  std::set<int> uniques_(clus_alloc.data(), clus_alloc.data() + ndata);
  std::vector<int> uniques(uniques_.begin(), uniques_.end());
  std::map<int, int> old2new;
  for (int i = 0; i < uniques.size(); i++) old2new.insert({uniques[i], i});

  for (int i = 0; i < ndata; i++) {
    clus_alloc(i) = old2new[clus_alloc(i)];
  }

  obs_by_clus.resize(0);
  obs_by_clus.resize(a_means.rows());
  etas_by_clus.resize(0);
  etas_by_clus.resize(a_means.rows());
  for (int i = 0; i < ndata; i++) {
    obs_by_clus[clus_alloc(i)].push_back(i);
    etas_by_clus[clus_alloc(i)].push_back(etas.row(i).transpose());
  }

  return;
}


void MultivariateConditionalMCMC::sample_etas() {

    MatrixXd M0(Lambda.transpose() * sigma_bar.asDiagonal());
    MatrixXd M1( M0 * Lambda);
    std::vector<MatrixXd> Sn_bar(a_means.rows());
    // type LLT for solving systems of equations
    std::vector<LLT<MatrixXd>> Sn_bar_cho (a_means.rows());

    for (int i=0; i < a_means.rows(); i++){
      Sn_bar[i]=M1+a_deltas[i].get_prec();
      Sn_bar_cho[i]= LLT<MatrixXd>(Sn_bar[i]);
    }

    MatrixXd M2(M0*data.transpose());
    MatrixXd G(ndata,dim_fact);

    // known terms of systems is depending on the single data
    for (int i=0; i < a_means.rows(); i++){
      MatrixXd B(dim_fact,obs_by_clus[i].size());
      B = (a_deltas[i].get_prec() * a_means.row(i).transpose()).replicate(1,B.cols());
      for (int j = 0; j < obs_by_clus[i].size(); j++){
        B.col(j) += M2.col(obs_by_clus[i][j]);
      }
      //B +=M2(all,obs_by_clus[i]);
      // each modified row has solution for points in the cluster.
      MatrixXd sol((Sn_bar_cho[i].solve(B)).transpose());
      for (int j = 0; j < obs_by_clus[i].size(); j++){
        G.row(obs_by_clus[i][j]) = sol.row(j);
      }
      //G(obs_by_clus[i],all)=(Sn_bar_cho[i].solve(B)).transpose();
    }

    // Here, G contains (in each row) mean of full-cond, while precisions have to be taken from Sn_bar
    // Now, I sample each eta from the full-cond multi-normal
    //#pragma omp parallel for
    for (int i=0; i < ndata; i++){
      etas.row(i)=multi_normal_prec_rng(G.row(i).transpose(), Sn_bar[clus_alloc(i)], Rng::Instance().get());
    }
    return;
}


void MultivariateConditionalMCMC::sample_sigma_bar() {

  //std::cout<<"sample sigma bar"<<std::endl;
  VectorXd betas = VectorXd::Constant(dim_data, _b_gamma);
  betas+=0.5*data.colwise().squaredNorm().transpose() + 0.5*(etas*Lambda.transpose()).colwise().squaredNorm().transpose();
  //#pragma omp parallel for
  for (int j=0; j < dim_data; j++){
    betas(j)-= Lambda.row(j)*etas.transpose()*data.col(j);
  }

  sigma_bar = Map<VectorXd>( gamma_rng(std::vector<double>(dim_data, ndata/2.0 + _a_gamma),
            std::vector<double>(betas.data(),betas.data()+betas.size()), Rng::Instance().get()).data(), dim_data);

  return;

}


void MultivariateConditionalMCMC::sample_Psi() {
//  std::cout<<"sample Psi"<<std::endl;
  // make it parallel omp
  //#pragma omp parallel for
  for (int j=0; j< dim_data; j++)
    for(int h=0; h< dim_fact; h++)
      Psi(j,h)=GIG::rgig(0.5, std::pow(Lambda(j,h)/(Phi(j,h)*tau),2), 1);

  return;
}


void MultivariateConditionalMCMC::sample_tau() {
  //std::cout<<"sample tau"<<std::endl;
  tau = GIG::rgig(dim_data*dim_fact*(_a_phi-1), 2*(Lambda.array().abs()/Phi.array()).sum() , 1);
  return;
}


void MultivariateConditionalMCMC::sample_Phi() {
//  std::cout<<"sample Phi"<<std::endl;
//#pragma omp parallel for
  for (int j=0; j < dim_data; j++)
    for (int h=0; h < dim_fact; h++)
      Phi(j,h)=GIG::rgig( _a_phi-1 , 2.0*std::abs(Lambda(j,h)), 1);

  Phi/= Phi.sum();
  return;
}

/*
VectorXd MultivariateConditionalMCMC::compute_grad_for_clus(
    int clus, const VectorXd &mean) {
  VectorXd grad = VectorXd::Zero(dim_data);
  for (const VectorXd datum : data_by_clus[clus]) grad += datum - mean;
  // TODO FIXME
  // grad += a_deltas[clus].get_prec() * (datum - mean);

  return grad;
}
*/



void MultivariateConditionalMCMC::get_state_as_proto(
    google::protobuf::Message *out_) {
  using namespace google::protobuf::internal;

  MultivariateMixtureState *out = down_cast<MultivariateMixtureState *>(out_);
  out->set_ma(a_means.rows());
  out->set_mna(na_means.rows());
  out->set_mtot(a_means.rows() + na_means.rows());

  for (int i = 0; i < a_means.rows(); i++) {
    EigenVector *mean;
    EigenMatrix *prec;
    mean = out->add_a_means();
    prec = out->add_a_deltas();

    to_proto(a_means.row(i).transpose(), mean);
    to_proto(a_deltas[i].get_prec(), prec);
  }

  for (int i = 0; i < na_means.rows(); i++) {
    EigenVector *mean;
    EigenMatrix *prec;
    mean = out->add_na_means();
    prec = out->add_na_deltas();

    to_proto(na_means.row(i).transpose(), mean);
    to_proto(na_deltas[i].get_prec(), prec);
  }

  to_proto(a_jumps, out->mutable_a_jumps());
  to_proto(na_jumps, out->mutable_na_jumps());

  *out->mutable_clus_alloc() = {clus_alloc.data(), clus_alloc.data() + ndata};

  out->set_u(u);

  for (int i = 0; i < ndata; i++) {
    EigenVector * eta;
    eta = out->add_etas();

    to_proto(etas.row(i).transpose(), eta);
  }

  to_proto(sigma_bar, out->mutable_sigma_bar());

  LambdaBlock lb;
  lb.set_tau(tau);
  to_proto(Phi, lb.mutable_phi());
  to_proto(Psi, lb.mutable_psi());
  to_proto(Lambda, lb.mutable_lamb());
  out->mutable_lamb_block()->CopyFrom(lb);

  return;


}

void MultivariateConditionalMCMC::print_debug_string() {
  std::cout << "*********** DEBUG STRING***********" << std::endl;
  std::cout << "#### ACTIVE: Number actives: " << a_means.rows() << std::endl;

  for (int i = 0; i < a_means.rows(); i++) {
    std::cout << "Component: " << i << ", weight: " << a_jumps(i)
              << ", mean: " << a_means.row(i) << ", precision: " << a_deltas[i]
              << std::endl;

    std::cout << std::endl;
  }

  std::cout << "#### NON - ACTIVE: Number actives: " << na_means.rows()
            << std::endl;

  for (int i = 0; i < na_means.rows(); i++) {
    std::cout << "Component: " << i << "weight: " << na_jumps(i)
              << ", mean: " << na_means.row(i) << ", precision: " << na_deltas[i]
              <<std::endl;
  }

  std::cout << std::endl;

  std::cout << "#### LAMBDA: " << Lambda << std::endl;
  std::cout << "#### tau: " << tau << std::endl;
  std::cout << "#### Psi: " << Psi << std::endl;
  std::cout << "#### Phi: " << Phi << std::endl;

  std::cout << "#### Allocations and  ETAS by cluster: " << std::endl;
  std::vector<int> aux(dim_fact);
  std::iota(aux.begin(),aux.end(),0);

  for (int i=0; i < a_means.rows() ; i++ ){
    std::cout<<"Cluster "<<i<<std::endl;
    std::cout<<"observations: "<<std::endl;
    for (auto j : obs_by_clus[i] ){
      std::cout<<j<<std::endl;
    }
  }

  std::cout<<"#### SIGMA BAR: "<<sigma_bar<<std::endl;

}


double MultivariateConditionalMCMC::a_means_acceptance_rate() {
        return sample_means_obj->Means_acc_rate();
}


double MultivariateConditionalMCMC::Lambda_acceptance_rate() {
      return sample_lambda->Lambda_acc_rate();
}

double MultivariateConditionalMCMC::get_ln_dens_ad() {
      return sample_lambda->get_ln_dens_ad();
}

double MultivariateConditionalMCMC::get_ln_dens_analytic() {
      return sample_lambda->get_ln_dens_analytic();
}

const MatrixXd& MultivariateConditionalMCMC::get_grad_log_ad() {
      return sample_lambda->get_grad_log_ad();
}

const MatrixXd& MultivariateConditionalMCMC::get_grad_log_analytic() {
      return sample_lambda->get_grad_log_analytic();
}

void MultivariateConditionalMCMC::print_data_by_clus(int clus) {
  for (const int &d : obs_by_clus[clus])
    std::cout << data.row(d).transpose() << std::endl;
}

}
