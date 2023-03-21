#include "alloc_means_sampler.hpp"
#include "conditional_mcmc.hpp"

namespace MCMCsampler {
double BaseMeansSampler::Means_acc_rate(){
        return (1.0 * acc_sampled_a_means) / (1.0 * tot_sampled_a_means);
}

//////////////////////////////////////
/// MeansSamplerClassic /////////
///////////////////////////////

void MeansSamplerClassic::perform_update_allocated(MatrixXd& Ctilde) {
  MatrixXd allmeans = mcmc->get_all_means();

  for (int i = 0; i < mcmc->get_num_a_means(); i++) {
    tot_sampled_a_means += 1;

    double currlik, proplik, prior_ratio, lik_ratio, arate;
    const VectorXd &currmean = mcmc->get_single_a_mean(i).transpose();
    const MatrixXd &cov_prop = MatrixXd::Identity(mcmc->get_dim_fact(), mcmc->get_dim_fact()) * prop_means_sigma;

    // we PROPOSE a new point from a multivariate normal, with mean equal to the current point
    // and covariance matrix diagonal

    VectorXd prop =
        stan::math::multi_normal_rng(currmean, cov_prop, Rng::Instance().get());

    if (mcmc->is_inside(prop)){ // if not, just keep the current mean and go to the next a_mean

        currlik = mcmc->lpdf_given_clus_multi(mcmc->get_etas_by_clus(i), currmean, mcmc->get_single_a_delta(i));
        proplik = mcmc->lpdf_given_clus_multi(mcmc->get_etas_by_clus(i), prop, mcmc->get_single_a_delta(i));

        lik_ratio = proplik - currlik;

        MatrixXd others(allmeans.rows() - 1, mcmc->get_dim_fact());
        others = delete_row(allmeans, i);
        MatrixXd others_trans = ((mcmc->pp_mix->get_A()*others.transpose()).colwise() + mcmc->pp_mix->get_b()).transpose();
        VectorXd prop_trans = mcmc->pp_mix->get_A() * prop + mcmc->pp_mix->get_b();

        MatrixXd Ctilde_oth(Ctilde);
        delete_row(&Ctilde_oth, i);
        delete_column(&Ctilde_oth,i);

        int n = Ctilde_oth.rows();

        // compute the row/column to be added to Ctilde: note that first i elements are over the diagonal, the last n-i
        // are below the diagonal.
        VectorXd col_new = VectorXd::Zero(n);
        const MatrixXd& Kappas_red=mcmc->pp_mix->get_kappas_red();
        for (int l = 0; l < n; l++) {
            //RowVectorXd vec(others_trans.row(l)-prop_trans.transpose());
            VectorXd vec_c(others_trans.row(l)-prop_trans.transpose());

            //int nthreads;
            //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
            /*for (int kind = 1; kind < Kappas_red.rows(); kind++) {
              //nthreads = omp_get_num_threads();
              //printf("Number of threads = %d\n", nthreads);
              double dotprod = Kappas_red.row(kind).dot(vec);
              col_new(l) += 2. * mcmc->pp_mix->get_phi_tildes_red()[kind] * std::cos(2. * stan::math::pi() * dotprod);
            }*/
            VectorXd dotprods = Kappas_red.bottomRows(Kappas_red.rows()-1) * vec_c;
            col_new(l) = 2.0 * (mcmc->pp_mix->get_phi_tildes_red().bottomRows(Kappas_red.rows()-1).array() * (2. * stan::math::pi() * dotprods).array().cos()).sum();

        }
        col_new.array() += mcmc->pp_mix->get_phi_tildes_red()[0];
        // Ctilde_prop differs from Ctilde only in the i row and column (diagonal element doesn't change)
        MatrixXd Ctilde_prop = Ctilde;
        Ctilde_prop.block(0,i,i,1) = col_new.head(i);
        Ctilde_prop.block(i+1,i,n-i,1) = col_new.tail(n-i);
        Ctilde_prop.block(i,0,1,i) = col_new.head(i).transpose();
        Ctilde_prop.block(i,i+1,1,n-i) = col_new.tail(n-i).transpose();

        prior_ratio =
            mcmc->pp_mix->papangelou(Ctilde_oth, Ctilde_prop) - mcmc->pp_mix->papangelou(Ctilde_oth, Ctilde);

        arate = lik_ratio + prior_ratio;

        if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < arate) {
          mcmc->set_single_a_mean(i, prop);
          // correcting the mistake of conditioning
          allmeans.row(i)=prop;
          acc_sampled_a_means += 1;
          Ctilde = Ctilde_prop;
        }
    }

  }
  return;
}




void MeansSamplerClassic::perform_update_trick_na(MatrixXd& Ctilde) {

  MatrixXd allmeans = mcmc->get_all_means();
  int num_alloc_m = mcmc->get_num_a_means();

  for (int i = 0; i < mcmc->get_num_na_means(); i++) {
    //tot_sampled_a_means += 1;

    double prior_ratio;
    const VectorXd &currmean = mcmc->get_single_na_mean(i).transpose();
    const MatrixXd &cov_prop = MatrixXd::Identity(mcmc->get_dim_fact(), mcmc->get_dim_fact()) * prop_means_sigma;

    // we PROPOSE a new point from a multivariate normal, with mean equal to the current point
    // and covariance matrix diagonal
    VectorXd prop =
        stan::math::multi_normal_rng(currmean, cov_prop, Rng::Instance().get());

    if (mcmc->is_inside(prop)){ // if not, just keep the current mean and go to the next a_mean

        MatrixXd others(allmeans.rows() - 1, mcmc->get_dim_fact());
        others = delete_row(allmeans, i+num_alloc_m);
        MatrixXd others_trans = ((mcmc->pp_mix->get_A()*others.transpose()).colwise() + mcmc->pp_mix->get_b()).transpose();
        VectorXd prop_trans = mcmc->pp_mix->get_A() * prop + mcmc->pp_mix->get_b();

        MatrixXd Ctilde_oth(Ctilde);
        delete_row(&Ctilde_oth, i+num_alloc_m);
        delete_column(&Ctilde_oth,i+num_alloc_m);

        int n = Ctilde_oth.rows();

        // compute the row/column to be added to Ctilde: note that first i elements are over the diagonal, the last n-i
        // are below the diagonal.
        VectorXd col_new = VectorXd::Zero(n);
        const MatrixXd& Kappas_red=mcmc->pp_mix->get_kappas_red();
        for (int l = 0; l < n; l++) {
            //RowVectorXd vec(others_trans.row(l)-prop_trans.transpose());
            VectorXd vec_c(others_trans.row(l)-prop_trans.transpose());

            //int nthreads;
            //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
            /*for (int kind = 1; kind < Kappas_red.rows(); kind++) {
              //nthreads = omp_get_num_threads();
              //printf("Number of threads = %d\n", nthreads);
              double dotprod = Kappas_red.row(kind).dot(vec);
              col_new(l) += 2. * mcmc->pp_mix->get_phi_tildes_red()[kind] * std::cos(2. * stan::math::pi() * dotprod);
            }*/
            VectorXd dotprods = Kappas_red.bottomRows(Kappas_red.rows()-1) * vec_c;
            col_new(l) = 2.0 * (mcmc->pp_mix->get_phi_tildes_red().bottomRows(Kappas_red.rows()-1).array() * (2. * stan::math::pi() * dotprods).array().cos()).sum();

        }
        col_new.array() += mcmc->pp_mix->get_phi_tildes_red()[0];
        // Ctilde_prop differs from Ctilde only in the i row and column (diagonal element doesn't change)
        MatrixXd Ctilde_prop = Ctilde;
        Ctilde_prop.block(0,i+ num_alloc_m,i+ num_alloc_m,1) = col_new.head(i+ num_alloc_m);
        Ctilde_prop.block(i+1+ num_alloc_m,i+ num_alloc_m,n-i- num_alloc_m,1) = col_new.tail(n-i- num_alloc_m);
        Ctilde_prop.block(i+ num_alloc_m,0,1,i+ num_alloc_m) = col_new.head(i+ num_alloc_m).transpose();
        Ctilde_prop.block(i+ num_alloc_m,i+1+ num_alloc_m,1,n-i- num_alloc_m) = col_new.tail(n-i- num_alloc_m).transpose();

        prior_ratio =
            mcmc->pp_mix->papangelou(Ctilde_oth, Ctilde_prop) - mcmc->pp_mix->papangelou(Ctilde_oth, Ctilde);

        if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < prior_ratio) {
          mcmc->set_single_na_mean(i, prop);
          //acc_sampled_a_means += 1;
          allmeans.row(i+ num_alloc_m)=prop;
          Ctilde = Ctilde_prop;
        }
    }

  }
  return;
}

//////////////////////////////////////
/// MeansSamplerMala /////////
///////////////////////////////


void MeansSamplerMala::perform_update_allocated(MatrixXd& Ctilde) {

  allmeans = mcmc->get_all_means();

  for (int i = 0; i < mcmc->get_num_a_means(); i++) {
    ind_mean = i;
    tot_sampled_a_means += 1;

    double ln_px_curr;
    VectorXd grad_ln_px_curr;
    VectorXd a_mean_curr = allmeans.row(i).transpose();

    stan::math::gradient(alloc_means_tar_fun, a_mean_curr, ln_px_curr, grad_ln_px_curr);

    // Proposal according MALA
    VectorXd prop_a_mean = a_mean_curr + mala_p_means*grad_ln_px_curr +
                      std::sqrt(2*mala_p_means)*Map<VectorXd>(normal_rng(std::vector<double>(mcmc->get_dim_fact(), 0.),
                    std::vector<double>(mcmc->get_dim_fact(),1.), Rng::Instance().get()).data() , mcmc->get_dim_fact());

    if (mcmc->is_inside(prop_a_mean)){ // if not, just keep the current mean and go to the next a_mean
      double ln_px_prop;
      VectorXd grad_ln_px_prop;
      stan::math::gradient(alloc_means_tar_fun, prop_a_mean, ln_px_prop, grad_ln_px_prop);

      // DEBUG
    // std::cout<<"Proposal Lambda: \n"<<prop_lambda<<std::endl;
    //  std::cout<<"Proposed Lambda"<<std::endl;

      // COMPUTE ACCEPTANCE PROBABILITY
      // (log) TARGET DENSITY TERMS
      double ln_ratio_target;
      ln_ratio_target = ln_px_prop - ln_px_curr;
      // DEBUG
      //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

      //(log) PROPOSAL DENSITY TERMS
      double ln_q_num, ln_q_den;
      ln_q_num = - (a_mean_curr - prop_a_mean - mala_p_means*grad_ln_px_prop).squaredNorm() /(4*mala_p_means);
      ln_q_den = - (prop_a_mean - a_mean_curr - mala_p_means*grad_ln_px_curr).squaredNorm() /(4*mala_p_means);

      // -> acceptance probability
      double ln_ratio = ln_ratio_target + ln_q_num - ln_q_den;

      if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < ln_ratio){
        //ACCEPTED
        acc_sampled_a_means += 1;
        mcmc->set_single_a_mean(i, prop_a_mean);
        //std::cout<<"accepted Lambda"<<std::endl;
      }
    }
  }
  return;
}


void MeansSamplerMala::perform_update_trick_na(MatrixXd& Ctilde) {

  allmeans = mcmc->get_all_means_reverse();

  for (int i = 0; i < mcmc->get_num_na_means(); i++) {
    ind_mean = i;
    //tot_sampled_a_means += 1;

    double ln_px_curr;
    VectorXd grad_ln_px_curr;
    VectorXd na_mean_curr = allmeans.row(i).transpose();
  //std::cout<<"before gradient"<<std::endl;
    stan::math::gradient(trick_na_means_tar_fun, na_mean_curr, ln_px_curr, grad_ln_px_curr);
    //std::cout<<"after gradient"<<std::endl;
    // Proposal according MALA
    VectorXd prop_na_mean = na_mean_curr + mala_p_means*grad_ln_px_curr +
                      std::sqrt(2*mala_p_means)*Map<VectorXd>(normal_rng(std::vector<double>(mcmc->get_dim_fact(), 0.),
                    std::vector<double>(mcmc->get_dim_fact(),1.), Rng::Instance().get()).data() , mcmc->get_dim_fact());

    if (mcmc->is_inside(prop_na_mean)){ // if not, just keep the current mean and go to the next a_mean
      double ln_px_prop;
      VectorXd grad_ln_px_prop;
      stan::math::gradient(trick_na_means_tar_fun, prop_na_mean, ln_px_prop, grad_ln_px_prop);

      // DEBUG
    // std::cout<<"Proposal Lambda: \n"<<prop_lambda<<std::endl;
    //  std::cout<<"Proposed Lambda"<<std::endl;

      // COMPUTE ACCEPTANCE PROBABILITY
      // (log) TARGET DENSITY TERMS
      double ln_ratio_target;
      ln_ratio_target = ln_px_prop - ln_px_curr;
      // DEBUG
      //std::cout<<"log_ratio: "<<log_ratio<<std::endl;

      //(log) PROPOSAL DENSITY TERMS
      double ln_q_num, ln_q_den;
      ln_q_num = - (na_mean_curr - prop_na_mean - mala_p_means*grad_ln_px_prop).squaredNorm() /(4*mala_p_means);
      ln_q_den = - (prop_na_mean - na_mean_curr - mala_p_means*grad_ln_px_curr).squaredNorm() /(4*mala_p_means);

      // -> acceptance probability
      double ln_ratio = ln_ratio_target + ln_q_num - ln_q_den;

      if (std::log(uniform_rng(0, 1, Rng::Instance().get())) < ln_ratio){
        //ACCEPTED
        //acc_sampled_a_means += 1;
        mcmc->set_single_na_mean(i, prop_na_mean);
        //std::cout<<"accepted Lambda"<<std::endl;
      }
    }
  }
  return;
}


}
