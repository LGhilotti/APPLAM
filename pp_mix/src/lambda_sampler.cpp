#include "lambda_sampler.hpp"
#include "conditional_mcmc.hpp"

namespace MCMCsampler {
double BaseLambdaSampler::Lambda_acc_rate(){
        return (1.0 * acc_sampled_Lambda) / (1.0 * tot_sampled_Lambda);
}

//////////////////////////////////////
/// LambdaSamplerClassic /////////
///////////////////////////////

double LambdaSamplerClassic::compute_exp_lik(const MatrixXd& lamb) const {

  return (mcmc->get_sigma_bar().array().sqrt().matrix().asDiagonal()*(mcmc->get_data().transpose() - lamb*mcmc->get_etas().transpose())).colwise().squaredNorm().sum();

}

double LambdaSamplerClassic::compute_exp_prior(const MatrixXd& lamb) const {

  return (lamb.array().square()/(mcmc->get_Psi().array()*mcmc->get_Phi().array().square())).sum() * (- 0.5 / (mcmc->get_tau()*mcmc->get_tau()));

}


void LambdaSamplerClassic::perform(MatrixXd& Ctilde) {
  //std::cout<<"sample Lambda"<<std::endl;
  // Current Lambda (here are the means) are expanded to vector<double> column major
  MatrixXd prop_lambda = Map<MatrixXd>(normal_rng( std::vector<double>(mcmc->get_Lambda().data(), mcmc->get_Lambda().data() + mcmc->get_dim_data()*mcmc->get_dim_fact()) ,
              std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(), prop_lambda_sigma), Rng::Instance().get()).data() , mcmc->get_dim_data(), mcmc->get_dim_fact());
  // DEBUG
  //std::cout<<"Proposal Lambda: \n"<<prop_lambda<<std::endl;
//  std::cout<<"Proposed Lambda"<<std::endl;

  tot_sampled_Lambda += 1;
  // we use log for each term
  double curr_lik, prop_lik;
  curr_lik = -0.5 * compute_exp_lik(mcmc->get_Lambda());
  prop_lik = -0.5 * compute_exp_lik(prop_lambda);
  // DEBUG
  //std::cout<<"curr_lik = "<<curr_lik<<"  ; prop_lik = "<<prop_lik<<std::endl;

  double curr_prior_cond_process, prop_prior_cond_process;
  MatrixXd means(mcmc->get_num_a_means()+mcmc->get_num_na_means(),mcmc->get_dim_fact());
  means << mcmc->get_a_means(), mcmc->get_na_means();

  mcmc->pp_mix->decompose_proposal(prop_lambda);

  curr_prior_cond_process = mcmc->pp_mix->dens_cond(means, true);
  prop_prior_cond_process = mcmc->pp_mix->dens_cond_in_proposal(means, true);
  // DEBUG
  //std::cout<<"curr_p_c_p = "<<curr_prior_cond_process<<"  ; prop_p_c_p = "<<prop_prior_cond_process<<std::endl;

  double curr_prior_lambda, prop_prior_lambda;
  curr_prior_lambda = compute_exp_prior(mcmc->get_Lambda());
  prop_prior_lambda = compute_exp_prior(prop_lambda);
  // DEBUG
 //std::cout<<"curr_prior_lambda = "<<curr_prior_lambda<<" ; prop_prior_lambda = "<<prop_prior_lambda<<std::endl;

  double curr_dens, prop_dens, log_ratio;
  curr_dens = curr_lik + curr_prior_cond_process + curr_prior_lambda;
  prop_dens = prop_lik + prop_prior_cond_process + prop_prior_lambda;
  // DEBUG
//  std::cout<<"curr_dens = "<<curr_dens<<" ; prop_dens = "<<prop_dens<<std::endl;

  log_ratio = prop_dens - curr_dens;
  // DEBUG
  //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

  if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < log_ratio){
    //ACCEPTED
    acc_sampled_Lambda += 1;
    mcmc->pp_mix->decompose_proposal(prop_lambda);
    //Lambda.swap(prop_lambda);
    //mcmc->Lambda = prop_lambda;
    mcmc->set_Lambda(prop_lambda);
    mcmc->pp_mix->update_decomposition_from_proposal();
    //std::cout<<"accepted Lambda"<<std::endl;
  }

  return;
}


//////////////////////////////////////
/// LambdaSamplerMala /////////
///////////////////////////////
/******* OLD COMPUTE GRAD ANALYIC
MatrixXd LambdaSamplerMala::compute_grad_analytic(){

  MatrixXd grad_log = MatrixXd::Zero(mcmc->get_dim_data(),mcmc->get_dim_fact());
  //First term

  MatrixXd A_tmp = mcmc->get_data().transpose() - mcmc->get_Lambda()*(mcmc->get_etas().transpose());
  for (int i =0; i<mcmc->get_ndata();i++){
    grad_log += A_tmp.col(i)*mcmc->get_etas().row(i);
  }
  grad_log = mcmc->get_sigma_bar().asDiagonal() * grad_log;

  //Second term

  int n_means=mcmc->get_num_a_means()+mcmc->get_num_na_means();
  MatrixXd mu_trans(n_means,mcmc->get_dim_fact());
  for (int i = 0; i < n_means; i++){
        mu_trans.row(i) = (mcmc->pp_mix->get_A() * mcmc->get_all_means().row(i).transpose() + mcmc->pp_mix->get_b()).transpose();
  }

  MatrixXd Ctilde(mu_trans.rows(), mu_trans.rows());
  MatrixXd Kappas (mcmc->pp_mix->get_kappas());
  VectorXd Phis (mcmc->pp_mix->get_phis());

  for (int l = 0; l < mu_trans.rows(); l++) {
    for (int m = l; m < mu_trans.rows(); m++) {
      double aux = 0.0;
      RowVectorXd vec(mu_trans.row(l)-mu_trans.row(m));
      //int nthreads;
      //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        //nthreads = omp_get_num_threads();
        //printf("Number of threads = %d\n", nthreads);
        double dotprod = Kappas.row(kind).dot(vec);
        aux += mcmc->pp_mix->get_phi_tildes()[kind] * std::cos(2. * stan::math::pi() * dotprod);
      }
      Ctilde(l, m) = aux;
      if (l!=m) Ctilde(m,l) = aux;
    }
  }

  // *** BEGIN TEST
  // - GRAD D_U
  /*
  MatrixXd Kappas (mcmc->pp_mix->get_kappas());
  VectorXd Phis (mcmc->pp_mix->get_phis());
  int d =  mcmc->get_dim_fact();
  const MatrixXd& lamb=mcmc->get_Lambda();
  LLT<MatrixXd> l_t_l (lamb.transpose() * lamb);
  MatrixXd part_g (2.0 * pow(l_t_l.matrixL().determinant(),2.0/d) * lamb);
  std::cout<<"part_g: \n"<<part_g<<std::endl;
  for (int kind=0; kind< Kappas.rows(); kind++) {
    VectorXd sol(l_t_l.solve(Kappas.row(kind).transpose()));
    std::cout<<"sol: \n"<<sol<<std::endl;
    MatrixXd s_part_g =MatrixXd::Constant(d,d, 1.0/d * Kappas.row(kind).dot(sol.transpose()) ) - Kappas.row(kind).transpose()*sol.transpose();
    std::cout<<"s_part_g: \n"<<s_part_g<<std::endl;
    MatrixXd gk (part_g *l_t_l.solve(s_part_g));
    std::cout<<"gk: \n"<<gk<<std::endl;

    grad_log += Phis[kind]/(1-Phis[kind]) * gk;
    std::cout<<"grad_log: \n"<<grad_log<<std::endl;

  }
  grad_log = 2*std::pow(stan::math::pi(),2.0)*std::pow(mcmc->pp_mix->get_c(),-2.0/d)*grad_log;

  // HERE, THE TEST FOR THE B MATRIX (look overleaf in the steps)
  LLT<MatrixXd> Ctil (Ctilde);

  int d =  mcmc->get_dim_fact();
  const MatrixXd& lamb=mcmc->get_Lambda();
  LLT<MatrixXd> l_t_l (lamb.transpose() * lamb);
  //MatrixXd l_t_l_inv (l_t_l.inverse());
  MatrixXd part_g (2.0 * pow(l_t_l.matrixL().determinant(),2.0/d) * lamb);
  //Redefine Kappas keeping only the ones with positive or 0 first component
  Kappas = Kappas.bottomRows(Kappas.rows()/(2*mcmc->pp_mix->get_N() +1) * (mcmc->pp_mix->get_N() +1));
  Phis = Phis.bottomRows(Phis.rows()/(2*mcmc->pp_mix->get_N() +1) * (mcmc->pp_mix->get_N() +1));
  MatrixXd SecTerm = MatrixXd::Zero(mcmc->get_dim_data(),d);
  for (int kind=0; kind< Kappas.rows(); kind++) {
    //construct g^k , u_k (real and img)
    VectorXd sol(l_t_l.solve(Kappas.row(kind).transpose()));
    // s_part_g contains the entire squared bracket
    MatrixXd s_part_g =MatrixXd::Constant(d,d, 1.0/d * Kappas.row(kind).dot(sol.transpose()) ) - Kappas.row(kind).transpose()*sol.transpose();
    // gk is the matrix g^k
    MatrixXd gk (part_g *l_t_l.solve(s_part_g));

    //define u_k
    VectorXd arg (2*stan::math::pi()* mu_trans * Kappas.row(kind).transpose());
    VectorXd uR (arg.array().cos());
    VectorXd uI (arg.array().sin());
    // scal is the scalar after g^k
    double scal = Phis[kind]/std::pow(1-Phis[kind],2.0)*(Ctil.solve(uR).dot(uR)+Ctil.solve(uI).dot(uI));

    SecTerm += scal * gk;
  }

  grad_log += -4*std::pow(stan::math::pi(),2.0)*std::pow(mcmc->pp_mix->get_c(),-2.0/d) * SecTerm;

  // *** END TEST
/*
  LLT<MatrixXd> Ctil (Ctilde);

  int d =  mcmc->get_dim_fact();
  const MatrixXd& lamb=mcmc->get_Lambda();
  LLT<MatrixXd> l_t_l (lamb.transpose() * lamb);
  //MatrixXd l_t_l_inv (l_t_l.inverse());
  MatrixXd part_g (2.0 * pow(l_t_l.matrixL().determinant(),2.0/d) * lamb);
  //Redefine Kappas keeping only the ones with positive or 0 first component
  Kappas = Kappas.bottomRows(Kappas.rows()/(2*mcmc->pp_mix->get_N() +1) * (mcmc->pp_mix->get_N() +1));
  Phis = Phis.bottomRows(Phis.rows()/(2*mcmc->pp_mix->get_N() +1) * (mcmc->pp_mix->get_N() +1));
  MatrixXd SecTerm = MatrixXd::Zero(mcmc->get_dim_data(),d);
  for (int kind=0; kind< Kappas.rows(); kind++) {
    //construct g^k , u_k (real and img)
    VectorXd sol(l_t_l.solve(Kappas.row(kind).transpose()));
    // s_part_g contains the entire squared bracket
    MatrixXd s_part_g =MatrixXd::Constant(d,d, 1.0/d * Kappas.row(kind).dot(sol.transpose()) ) - Kappas.row(kind).transpose()*sol.transpose();
    // gk is the matrix g^k
    MatrixXd gk (part_g *l_t_l.solve(s_part_g));

    //define u_k
    VectorXd arg (2*stan::math::pi()* mu_trans * Kappas.row(kind).transpose());
    VectorXd uR (arg.array().cos());
    VectorXd uI (arg.array().sin());
    // scal is the scalar after g^k
    double scal = Phis[kind]/std::pow(1-Phis[kind],2.0)*(((1-Phis[kind])/(1-std::exp(-mcmc->pp_mix->get_Ds())))-Ctil.solve(uR).dot(uR)-Ctil.solve(uI).dot(uI));

    SecTerm += scal * gk;
  }

  grad_log += 4*std::pow(stan::math::pi(),2.0)*std::pow(mcmc->pp_mix->get_c(),-2.0/d) * SecTerm;

  //Third term
  grad_log -= (mcmc->get_Lambda().array() / (mcmc->get_Phi().array().square() * mcmc->get_Psi().array() * mcmc->get_tau()*mcmc->get_tau())).matrix();


  return grad_log;
}
*/

double LambdaSamplerMala::compute_ln_dens_analytic(const MatrixXd& lamb, double lndetCtil_prop){

  const MatrixXd& data = mcmc->get_data();
  const MatrixXd& etas = mcmc->get_etas();

  double ln_dens = 0.;
 // FIRST TERM

  VectorXd betas = VectorXd::Zero(lamb.rows());
  betas+=0.5*data.colwise().squaredNorm().transpose() + 0.5*(etas*lamb.transpose()).colwise().squaredNorm().transpose();
  //#pragma omp parallel for
  for (int j=0; j < lamb.rows(); j++){
    betas(j)-= lamb.row(j)*etas.transpose()*data.col(j);
  }

  ln_dens -= betas.dot(mcmc->get_sigma_bar());

  // SECOND TERM
  ln_dens += mcmc->pp_mix->dens_cond_in_proposal(mcmc->get_all_means(), lndetCtil_prop, true);

  // THIRD TERM
  ln_dens -= 0.5 * (lamb.array().square() / (mcmc->get_Phi().array().square() * mcmc->get_Psi().array() * mcmc->get_tau()*mcmc->get_tau())).sum();

  return ln_dens;
}


double LambdaSamplerMala::compute_ln_dens_analytic(double lndetCtil){

  const MatrixXd& lamb = mcmc->get_Lambda();
  const MatrixXd& data = mcmc->get_data();
  const MatrixXd& etas = mcmc->get_etas();

  double ln_dens = 0.;
 // FIRST TERM

  VectorXd betas = VectorXd::Zero(lamb.rows());
  betas+=0.5*data.colwise().squaredNorm().transpose() + 0.5*(etas*lamb.transpose()).colwise().squaredNorm().transpose();
  //#pragma omp parallel for
  for (int j=0; j < lamb.rows(); j++){
    betas(j)-= lamb.row(j)*etas.transpose()*data.col(j);
  }

  ln_dens -= betas.dot(mcmc->get_sigma_bar());

  // SECOND TERM
  ln_dens += mcmc->pp_mix->dens_cond(mcmc->get_all_means(),lndetCtil, true);

  // THIRD TERM
  ln_dens -= 0.5 * (lamb.array().square() / (mcmc->get_Phi().array().square() * mcmc->get_Psi().array() * mcmc->get_tau()*mcmc->get_tau())).sum();

  return ln_dens;
}




MatrixXd LambdaSamplerMala::compute_grad_analytic(const MatrixXd& lamb, const MatrixXd& Ctilde){

  const VectorXd& Phis_red (mcmc->pp_mix->get_phis_tmp_red());
  double Ds_red = mcmc->pp_mix->get_Ds_tmp_red();

  return compute_gr_an(lamb,Ctilde,Phis_red, Ds_red);

}



MatrixXd LambdaSamplerMala::compute_grad_analytic(const MatrixXd& Ctilde){

  return compute_gr_an(mcmc->get_Lambda(), Ctilde, mcmc->pp_mix->get_phis_red(), mcmc->pp_mix->get_Ds_red());

}


MatrixXd LambdaSamplerMala::compute_gr_an(const MatrixXd& lamb, const MatrixXd& Ctilde, const VectorXd& Phis_red, double Ds_red){

  int d =  mcmc->get_dim_fact();

  MatrixXd grad_log = MatrixXd::Zero(mcmc->get_dim_data(),d);

  //First term

  MatrixXd A_tmp = mcmc->get_data().transpose() - lamb*(mcmc->get_etas().transpose());

  for (int i =0; i<mcmc->get_ndata();i++){
    grad_log += A_tmp.col(i)*mcmc->get_etas().row(i);
  }
  grad_log = mcmc->get_sigma_bar().asDiagonal() * grad_log;

  //Second term
  int n_means=mcmc->get_num_a_means()+mcmc->get_num_na_means();
  MatrixXd mu_trans(n_means,d);
  for (int i = 0; i < n_means; i++){
        mu_trans.row(i) = (mcmc->pp_mix->get_A() * mcmc->get_all_means().row(i).transpose() + mcmc->pp_mix->get_b()).transpose();
  }

  const MatrixXd& Kappas_red (mcmc->pp_mix->get_kappas_red());

  LLT<MatrixXd> Ctil (Ctilde);

  LLT<MatrixXd> l_t_l (lamb.transpose() * lamb);
  MatrixXd part_g (2.0 * pow(l_t_l.matrixL().determinant(),2.0/d) * lamb);
  //Redefine Kappas keeping only the ones with positive or 0 first component
  MatrixXd SecTerm = MatrixXd::Zero(mcmc->get_dim_data(),d);

  #pragma omp parallel
  {
    MatrixXd SecTerm_private = MatrixXd::Zero(mcmc->get_dim_data(),d);

    #pragma omp for nowait //fill SecTerm in parallel
      for (int kind=1; kind< Kappas_red.rows(); kind++) {
        //construct g^k , u_k (real and img)
        VectorXd sol(l_t_l.solve(Kappas_red.row(kind).transpose()));
        // s_part_g contains the entire squared bracket
        MatrixXd s_part_g =MatrixXd::Constant(d,d, 1.0/d * Kappas_red.row(kind).dot(sol.transpose()) ) - Kappas_red.row(kind).transpose()*sol.transpose();
        // gk is the matrix g^k
        MatrixXd gk (part_g *l_t_l.solve(s_part_g));

        //define u_k
        VectorXd arg (2*stan::math::pi()* mu_trans * Kappas_red.row(kind).transpose());
        VectorXd uR (arg.array().cos());
        VectorXd uI (arg.array().sin());
        // scal is the scalar after g^k
        double scal = Phis_red[kind]/std::pow(1-Phis_red[kind],2.0)*(((1-Phis_red[kind])/(1-std::exp(-Ds_red)))-Ctil.solve(uR).dot(uR)-Ctil.solve(uI).dot(uI));

        SecTerm_private += scal * gk;
      }

      #pragma omp critical
      SecTerm += SecTerm_private;
  }
  grad_log += 4*std::pow(stan::math::pi(),2.0)*std::pow(mcmc->pp_mix->get_c(),-2.0/d) * SecTerm;

  //Third term
  grad_log -= (lamb.array() / (mcmc->get_Phi().array().square() * mcmc->get_Psi().array() * mcmc->get_tau()*mcmc->get_tau())).matrix();


  return grad_log;
}



void LambdaSamplerMala::perform(MatrixXd& Ctilde) {

  /*********************************************************/
  /******************** AUTODIFF VERSION ****************/
  /********************************************************/

  // Current Lambda (here are the means) are expanded to vector<double> column major
/*  double ln_px_curr;
  VectorXd grad_ln_px_curr;
  const VectorXd Lambda_curr = Map<const VectorXd>(mcmc->get_Lambda().data(), mcmc->get_dim_data()*mcmc->get_dim_fact()); // column-major
 // THIS IS THE GRADIENT VIA AUTODIFF
  stan::math::gradient(lambda_tar_fun, Lambda_curr, ln_px_curr, grad_ln_px_curr);


  // Proposal according MALA
  VectorXd prop_lambda_vec = Lambda_curr + mala_p_lambda*grad_ln_px_curr +
                    std::sqrt(2*mala_p_lambda)*Map<VectorXd>(normal_rng(std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(), 0.),
                  std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(),1.), Rng::Instance().get()).data() , mcmc->get_dim_data()*mcmc->get_dim_fact());


  double ln_px_prop;
  VectorXd grad_ln_px_prop;
  stan::math::gradient(lambda_tar_fun, prop_lambda_vec, ln_px_prop, grad_ln_px_prop);

  MatrixXd prop_lambda = Map<MatrixXd>(prop_lambda_vec.data(), mcmc->get_dim_data(), mcmc->get_dim_fact());

  /****** TEST FOR SAME GRADIENTS AND DENSITY *****/
  //THIS IS THE GRADIENT VIA ANALYTICAL COMPUTATION
/*
  grad_log_analytic = compute_grad_analytic(Ctilde);
  double lndetCtil = 2.0 * std::log(Ctilde.llt().matrixL().determinant());

  mcmc->pp_mix->decompose_proposal(prop_lambda);
  MatrixXd Ctilde_prop = mcmc->pp_mix->compute_Ctilde_prop(mcmc->get_all_means());
  double lndetCtil_prop = 2.0 * std::log(Ctilde_prop.llt().matrixL().determinant());
  ln_dens_analytic = compute_ln_dens_analytic(lndetCtil) - compute_ln_dens_analytic(prop_lambda, lndetCtil_prop);

  ln_dens_ad = ln_px_curr - ln_px_prop;
*/

/*
  tot_sampled_Lambda += 1;
  // COMPUTE ACCEPTANCE PROBABILITY
  // (log) TARGET DENSITY TERMS
  double ln_ratio_target;
  ln_ratio_target = ln_px_prop - ln_px_curr;
  // DEBUG
  //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

  //(log) PROPOSAL DENSITY TERMS
  double ln_q_num, ln_q_den;
  ln_q_num = - (Lambda_curr - prop_lambda_vec - mala_p_lambda*grad_ln_px_prop).squaredNorm() /(4*mala_p_lambda);
  ln_q_den = - (prop_lambda_vec - Lambda_curr - mala_p_lambda*grad_ln_px_curr).squaredNorm() /(4*mala_p_lambda);

  // -> acceptance probability
  double ln_ratio = ln_ratio_target + ln_q_num - ln_q_den;

  if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < ln_ratio){
    //ACCEPTED
    acc_sampled_Lambda += 1;
    mcmc->pp_mix->decompose_proposal(prop_lambda);
    Ctilde = mcmc->pp_mix->compute_Ctilde_prop(mcmc->get_all_means());
    //Lambda.swap(prop_lambda);
    //mcmc->Lambda = prop_lambda;
    mcmc->set_Lambda(prop_lambda); // the pointer in Determ to Lambda tracks this modified Lambda.
    mcmc->pp_mix->update_decomposition_from_proposal();
    //std::cout<<"accepted Lambda"<<std::endl;
  }

  /*********************************************************/
  /******************** END AUTODIFF VERSION ****************/
  /********************************************************/


  /*********************************************************/
  /******************** ANALYTICAL VERSION ****************/
  /********************************************************/
  // test reduced
  //std::cout<<"dens_cond:\n"<<mcmc->pp_mix->dens_cond(mcmc->get_all_means(),true)<<std::endl;
  //std::cout<<"dens_cond_red:\n"<<mcmc->pp_mix->dens_cond_red(mcmc->get_all_means(),true)<<std::endl;
  //std::cout<<"Ds:\n"<<mcmc->pp_mix->get_Ds()<<std::endl;
  //std::cout<<"phi_tildes:\n"<<mcmc->pp_mix->get_phi_tildes()<<std::endl;
  // end test reduced

// Ctilde in mcmc is related to the current means and the current lambda.
// When proposing the new lambda, of course I need to compute Ctilde from scratch.
  //Given the mus, I can compute Ctilde of the current Lambda and current mus, so that
  // I use it both for density evaluation and gradient computation. The, DPP has two methods, if I pass the mus,
  // it computes Ctilde, if not uses Ctilde already set.
  double lndetCtil = 2.0 * std::log(Ctilde.llt().matrixL().determinant());
  double ln_px_curr (compute_ln_dens_analytic(lndetCtil) );
  //std::cout<<"ln_px_curr:\n"<<ln_px_curr<<std::endl;
  //THIS IS THE GRADIENT VIA ANALYTICAL COMPUTATION (in the current Lambda)
  MatrixXd grad_ln_px_curr (compute_grad_analytic(Ctilde) );
  //std::cout<<"grad_ln_px_curr:\n"<<grad_ln_px_curr<<std::endl;

  // Proposal according MALA
  MatrixXd prop_lambda = mcmc->get_Lambda() + mala_p_lambda*grad_ln_px_curr +
                    std::sqrt(2*mala_p_lambda)*Map<MatrixXd>(normal_rng(std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(), 0.),
                  std::vector<double>(mcmc->get_dim_data()*mcmc->get_dim_fact(),1.), Rng::Instance().get()).data() , mcmc->get_dim_data() , mcmc->get_dim_fact());

  mcmc->pp_mix->decompose_proposal(prop_lambda);
  MatrixXd Ctilde_prop = mcmc->pp_mix->compute_Ctilde_prop(mcmc->get_all_means());
  double lndetCtil_prop = 2.0 * std::log(Ctilde_prop.llt().matrixL().determinant());

  double ln_px_prop (compute_ln_dens_analytic(prop_lambda, lndetCtil_prop) );
  MatrixXd grad_ln_px_prop (compute_grad_analytic(prop_lambda,Ctilde_prop) );

  tot_sampled_Lambda += 1;
  // COMPUTE ACCEPTANCE PROBABILITY
  // (log) TARGET DENSITY TERMS
  double ln_ratio_target;
  ln_ratio_target = ln_px_prop - ln_px_curr;
  // DEBUG
  //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

  //(log) PROPOSAL DENSITY TERMS
  double ln_q_num, ln_q_den;
  ln_q_num = - (mcmc->get_Lambda() - prop_lambda - mala_p_lambda*grad_ln_px_prop).squaredNorm() /(4*mala_p_lambda);
  ln_q_den = - (prop_lambda - mcmc->get_Lambda() - mala_p_lambda*grad_ln_px_curr).squaredNorm() /(4*mala_p_lambda);

  // -> acceptance probability
  double ln_ratio = ln_ratio_target + ln_q_num - ln_q_den;

  if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < ln_ratio){
    //ACCEPTED
    acc_sampled_Lambda += 1;
    mcmc->set_Lambda(prop_lambda);
    mcmc->pp_mix->update_decomposition_from_proposal();
    //std::cout<<"accepted Lambda"<<std::endl;
    Ctilde = Ctilde_prop;
  }

  /*********************************************************/
  /******************** END ANALYTICAL VERSION ****************/
  /********************************************************/

  return;

}

}
