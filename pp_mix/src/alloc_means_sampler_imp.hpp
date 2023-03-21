#ifndef ALLOC_MEANS_SAMPLER_IMP
#define ALLOC_MEANS_SAMPLER_IMP

namespace MCMCsampler {

// returns the ln of full-cond of Lambda|rest in current Lambda (lamb is vectorized)
template <typename T>
T MeansSamplerMala::alloc_means_target_function::operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & mean) const {
/*
  using std::pow; using std::exp; using std::log;

  MatrixXd others(msm.get_num_allmeans() - 1, msm.mcmc->get_dim_fact());
  others = delete_row(msm.get_allmeans(), msm.get_ind_mean());
*/
  T output{0.};
/*
  int n = msm.mcmc->get_etas_by_clus(msm.get_ind_mean()).size();

  MatrixXd cluster_etas(msm.mcmc->get_dim_fact(), n );
  for (int i = 0; i < n ; i++)
  {
    cluster_etas.col(i) = msm.mcmc->get_etas_by_clus(msm.get_ind_mean())[i];
  }

  output -= 0.5 * sum(multiply(msm.mcmc->get_single_a_delta(msm.get_ind_mean()).get_cho_factor_eval(), (cluster_etas.colwise()-mean)).colwise().squaredNorm());

  MatrixXd others_trans(others.rows() ,msm.mcmc->get_dim_fact());
  for (int i = 0; i < others_trans.rows() ; i++){
        others_trans.row(i) = (msm.mcmc->pp_mix->get_A() * others.row(i).transpose() + msm.mcmc->pp_mix->get_b()).transpose();
  }

  Eigen::Matrix<T, 1, Eigen::Dynamic> mean_trans;
  mean_trans = (multiply(msm.mcmc->pp_mix->get_A(), mean) + msm.mcmc->pp_mix->get_b()).transpose();

  int n_all_means = msm.get_num_allmeans();
  Matrix<T,Dynamic,Dynamic> Ctilde(n_all_means, n_all_means);
  const MatrixXd& Kappas = msm.mcmc->pp_mix->get_kappas();
  VectorXd vec_phi_tildes = msm.mcmc->pp_mix->get_phi_tildes();

  for (int l = 0; l < others.rows(); l++) {
    for (int m = l; m < others.rows(); m++) {
      double aux = 0.0;
      RowVectorXd vec(others_trans.row(l) - others_trans.row(m));
      #pragma omp parallel for default(none) firstprivate(Kappas,vec, vec_phi_tildes) reduction(+:aux)
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        double dotprod = Kappas.row(kind).dot(vec);
        aux += vec_phi_tildes(kind) * std::cos(2. * stan::math::pi() * dotprod);
      }
      Ctilde(l, m) = aux;
      if (l!=m) Ctilde(m,l) = aux;
    }
  }
  for (int m = 0; m < others.rows(); m++) {
    T aux = 0.0;
    for (int kind = 0; kind < Kappas.rows(); kind++) {
        T dotprod = dot_product(Kappas.row(kind), mean_trans - others_trans.row(m));
        aux += vec_phi_tildes(kind) * cos(2. * stan::math::pi() * dotprod);
    }
    Ctilde(n_all_means-1, m) = aux;
    Ctilde(m, n_all_means-1) = aux;
  }

  Ctilde(n_all_means-1, n_all_means-1) = Ctilde(0,0);
  output += +log_determinant(Ctilde);
*/
  return output;

}




template <typename T>
T MeansSamplerMala::trick_na_means_target_function::operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & mean) const {
/*
  using std::pow; using std::exp; using std::log;

  MatrixXd others(msm.get_num_allmeans() - 1, msm.mcmc->get_dim_fact());
  others = delete_row(msm.get_allmeans(), msm.get_ind_mean());
*/
  T output{0.};
/*

  MatrixXd others_trans(others.rows(),msm.mcmc->get_dim_fact());
  for (int i = 0; i < others_trans.rows() ; i++){
        others_trans.row(i) = (msm.mcmc->pp_mix->get_A() * others.row(i).transpose() + msm.mcmc->pp_mix->get_b()).transpose();
  }
  Eigen::Matrix<T, 1, Eigen::Dynamic> mean_trans;
  mean_trans = (multiply(msm.mcmc->pp_mix->get_A(), mean) + msm.mcmc->pp_mix->get_b()).transpose();

  int n_all_means = msm.get_num_allmeans();
  Matrix<T,Dynamic,Dynamic> Ctilde(n_all_means, n_all_means);
  const MatrixXd& Kappas = msm.mcmc->pp_mix->get_kappas();
  VectorXd vec_phi_tildes = msm.mcmc->pp_mix->get_phi_tildes();

  for (int l = 0; l < others.rows(); l++) {
    for (int m = l; m < others.rows(); m++) {
      double aux = 0.0;
      RowVectorXd vec(others_trans.row(l) - others_trans.row(m));
      #pragma omp parallel for default(none) firstprivate(Kappas,vec, vec_phi_tildes) reduction(+:aux)
      for (int kind = 0; kind < Kappas.rows(); kind++) {
        double dotprod = Kappas.row(kind).dot(vec);
        aux += vec_phi_tildes(kind) * cos(2. * stan::math::pi() * dotprod);
      }
      Ctilde(l, m) = aux;
      if (l!=m) Ctilde(m,l) = aux;

    }
  }

  for (int m = 0; m < others.rows(); m++) {
    T aux = 0.0;
    for (int kind = 0; kind < Kappas.rows(); kind++) {
        T dotprod = dot_product(Kappas.row(kind), mean_trans - others_trans.row(m));
        aux += vec_phi_tildes(kind) * cos(2. * stan::math::pi() * dotprod);
    }
    Ctilde(n_all_means-1, m) = aux;
    Ctilde(m, n_all_means-1) = aux;
  }

  Ctilde(n_all_means-1, n_all_means-1) = Ctilde(0,0);
  output += +log_determinant(Ctilde);
*/
  return output;

}

};


#endif
