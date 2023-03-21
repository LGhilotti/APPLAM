#include "determinantalPP.hpp"

#include <numeric>

using stan::math::LOG_SQRT_PI;
double PI = stan::math::pi();


DeterminantalPP::DeterminantalPP(const MatrixXd &ranges, int N, double c, double s):
  ranges(ranges), N(N), c(c), s(s) {

  dim = ranges.cols();
  diff_range = (ranges.row(1) - ranges.row(0)).transpose();
  vol_range = diff_range.prod();

  A = MatrixXd::Zero(dim, dim);
  b = VectorXd::Zero(dim);
  for (int i = 0; i < dim; i++) {
    A(i, i) = 1.0 / (ranges(1, i) - ranges(0, i));
    b(i) = -A(i, i) * (ranges(1, i) + ranges(0, i)) / 2.0;
  }

  compute_Kappas();

  phis_red.resize(Kappas_red.rows());
  phi_tildes_red.resize(Kappas_red.rows());
  phis_tmp_red.resize(Kappas_red.rows());
  phi_tildes_tmp_red.resize(Kappas_red.rows());

  /*
  std::cout << "ranges: "<<this->ranges<<std::endl;
  std::cout << "N: "<<this->N<<std::endl;
  std::cout << "c: "<<this->c<<std::endl;
  std::cout << "s: "<<this->s<<std::endl;
  std::cout << "dim: "<<this->dim<<std::endl;
  std::cout << "diff_range: "<<this->diff_range<<std::endl;
  std::cout << "vol_range: "<<this->vol_range<<std::endl;
  std::cout<<" end Base constructor"<<std::endl;
  */
  return;
}


void DeterminantalPP::set_decomposition(const MatrixXd * lambda) {

  Lambda = lambda;
  //compute_eigen_and_cstar(&Ds, &phis, &phi_tildes, &c_star, lambda);
  compute_eigen_and_cstar_red(&Ds_red, &phis_red, &phi_tildes_red, &c_star_red, lambda);
  return;

}

/*
void DeterminantalPP::compute_eigen_and_cstar(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda){


  *D_ = 0.0;
  *C_star_ = 0.0;

  LLT<MatrixXd> M ((*lambda).transpose() * (*lambda));
  // compute determinant of Lambda^T Lambda
  double det = std::pow(M.matrixL().determinant(),2);

  double esp_fact = -2*std::pow(stan::math::pi(),2)*std::pow(det,1.0/dim)*std::pow(c,-2.0/dim);
  ArrayXd vec_phi(Kappas.rows());
  //#pragma omp parallel for default(none) shared(vec_phi, Kappas, s,esp_fact,M)
  for (int i = 0; i < Kappas.rows(); i++) {
    VectorXd sol = M.solve(Kappas.row(i).transpose());
    double dot_prod = (Kappas.row(i)).dot(sol);
    vec_phi(i) = s*std::exp(esp_fact*dot_prod);
  }

  *Phis_ = vec_phi.matrix();
  *Phi_tildes_ = (vec_phi / (1 - vec_phi)).matrix();

  *D_ = log(1 + (*Phi_tildes_).array()).sum();
  *C_star_ = (*Phi_tildes_).sum();

  return;

}*/

void DeterminantalPP::compute_eigen_and_cstar_red(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda){


  *D_ = 0.0;
  *C_star_ = 0.0;

  LLT<MatrixXd> M ((*lambda).transpose() * (*lambda));
  // compute determinant of Lambda^T Lambda
  double det = std::pow(M.matrixL().determinant(),2);

  double esp_fact = -2*std::pow(stan::math::pi(),2)*std::pow(det,1.0/dim)*std::pow(c,-2.0/dim);
//  ArrayXd vec_phi(Kappas.rows());
  //#pragma omp parallel for default(none) shared(vec_phi, Kappas, s,esp_fact,M)
  /*for (int i = 0; i < Kappas_red.rows(); i++) {
    VectorXd sol = M.solve(Kappas_red.row(i).transpose());
    double dot_prod = (Kappas_red.row(i)).dot(sol);
    (*Phis_)(i) = s*std::exp(esp_fact*dot_prod);
  }*/
  ArrayXd dotprods = (Kappas_red * M.solve(Kappas_red.transpose())).diagonal();
  *Phis_ = s*(esp_fact*dotprods).exp();

  *Phi_tildes_ = ((*Phis_).array() / (1 - (*Phis_).array())).matrix();

  *D_ = log(1 + (*Phi_tildes_).array()).sum() * 2.0 - log(1+(*Phi_tildes_)(0));
  *C_star_ = (*Phi_tildes_).sum() * 2.0 - (*Phi_tildes_)(0);

  return;

}


void DeterminantalPP::decompose_proposal(const MatrixXd& lambda) {

  //compute_eigen_and_cstar(&Ds_tmp, &phis_tmp, &phi_tildes_tmp, &c_star_tmp, &lambda);
  compute_eigen_and_cstar_red(&Ds_tmp_red, &phis_tmp_red, &phi_tildes_tmp_red, &c_star_tmp_red, &lambda);

  return;

}


void DeterminantalPP::update_decomposition_from_proposal() {
/*
  std::swap(Ds, Ds_tmp);
  phis.swap(phis_tmp);
  phi_tildes.swap(phi_tildes_tmp);
  std::swap(c_star, c_star_tmp);
*/
  std::swap(Ds_red, Ds_tmp_red);
  phis_red.swap(phis_tmp_red);
  phi_tildes_red.swap(phi_tildes_tmp_red);
  std::swap(c_star_red, c_star_tmp_red);
  return;
}

// compute just once the grid for summation over Z^dim
void DeterminantalPP::compute_Kappas() {

  std::vector<double> k(2 * N + 1);
  for (int n = -N; n <= N; n++) {
    k[n + N] = n;
  }
  std::vector<std::vector<double>> kappas_red;
  if (dim == 1) {
    kappas_red.resize(k.size()/2 +1);
    for (int i = 0; i < kappas_red.size(); i++) kappas_red[kappas_red.size() - 1-i].push_back(-k[i]);
  } else {
    kappas_red = cart_product(k, dim);
  }

  Kappas_red.resize(kappas_red.size(), dim);
  for (int i = 0; i < kappas_red.size(); i++) {
    Kappas_red.row(i) = Map<VectorXd>(kappas_red[i].data(), dim).transpose();
  }


  return;

}


double DeterminantalPP::dens_cond_in_proposal(const MatrixXd& x, double lndetCtil_prop, bool log) {

  double out = ln_dens_process(x, lndetCtil_prop, Ds_tmp_red, c_star_tmp_red);
  out -= std::log(1-std::exp(-Ds_tmp_red));

  if (!log) out=std::exp(out);

  return out;

}


double DeterminantalPP::dens_cond(const MatrixXd& x, double lndetCtil, bool log) {

  double out = ln_dens_process(x, lndetCtil, Ds_red, c_star_red);
  out -= std::log(1-std::exp(-Ds_red));

  if (!log) out = std::exp(out);

  return out;

}

double DeterminantalPP::ln_dens_process(const MatrixXd& x, double lndetCtil_p, double Ds_p,
   double c_star_p){

  double out;
  int n;

  // check if it's jut one point
  if ((x.size() == 1 && dim == 1) || (x.rows() == 1 & dim > 1) ||
      (x.cols() == 1 && dim > 1)) {
    n = 1;
    out =
        -1.0 * n * std::log(vol_range) - Ds_p + std::log(c_star_p);
  }
  else {
    int n = x.rows();
    bool check_range = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        if (x(i, j) < ranges(0, j) || x(i, j) > ranges(1, j))
          check_range = false;
      }
    }
    if (check_range) {
      out = -1.0 * n * std::log(vol_range) - Ds_p + lndetCtil_p;

    } else {
      out = stan::math::NEGATIVE_INFTY;
    }
  }
  // std::cout << "dens: " << out << std::endl;

  return out;

}


// this is computed wrt the Lambda matrix and the current decomposition (in phi,phi_tildes)
double DeterminantalPP::papangelou(const MatrixXd& Ctilde, const MatrixXd &Ctilde_xi, bool log) {
  // starting from Ctilde with the current allmeans, it adds one row/column and compute the Ctilde'.
  // Transform xi to be in the unit cube centered in 0
  double out = -1.0*std::log(vol_range)+ 2.0 * std::log(Ctilde_xi.llt().matrixL().determinant()) - 2.0 * std::log(Ctilde.llt().matrixL().determinant());

  //double out = -1.0*std::log(vol_range)+ log_det_Ctilde(alltrans, phi_tildes) - log_det_Ctilde(alltrans.topRows(n-1), phi_tildes);
  //std::cout<<"out: "<<out<<std::endl;
  if (!log) out = std::exp(out);

  return out;
}



MatrixXd DeterminantalPP::sample_uniform(int npoints) {
  MatrixXd out(npoints, dim);
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < npoints; i++) {
      out(i, j) =
          uniform_rng(ranges(0, j), ranges(1, j), Rng::Instance().get());
    }
  }

  return out;
}


VectorXd DeterminantalPP::phi_star_rng() {
  VectorXd out(dim);
  for (int i = 0; i < dim; i++) {
    out(i) = uniform_rng(ranges(0, i), ranges(1, i), Rng::Instance().get());
  }
  return out;
}

double DeterminantalPP::phi_star_dens(VectorXd xi, bool log) {
  double out = std::log(c_star_red) - std::log(vol_range);
  if (!log) out = std::exp(out);

  return out;
}

MatrixXd DeterminantalPP::compute_Ctilde(const MatrixXd& means){

  MatrixXd means_trans = ((A * means.transpose()).colwise() + b).transpose();
//  std::cout<<"means_trans:\n"<<means_trans<<std::endl;
  MatrixXd Ctilde(means_trans.rows(), means_trans.rows());

  for (int l = 0; l < means_trans.rows()-1; l++) {
    for (int m = l+1; m < means_trans.rows(); m++) {
      //double aux = 0.0;
      //RowVectorXd vec(means_trans.row(l)-means_trans.row(m));
      VectorXd vec_c(means_trans.row(l)-means_trans.row(m));

      //int nthreads;
      //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
      /*for (int kind = 1; kind < Kappas_red.rows(); kind++) {
        //nthreads = omp_get_num_threads();
        //printf("Number of threads = %d\n", nthreads);
        double dotprod = Kappas_red.row(kind).dot(vec);
        aux += phi_tildes_red[kind] * std::cos(2. * stan::math::pi() * dotprod);
      }*/
      VectorXd dotprods = Kappas_red.bottomRows(Kappas_red.rows()-1) * vec_c;
      double aux_c = (phi_tildes_red.bottomRows(Kappas_red.rows()-1).array() * (2. * stan::math::pi() * dotprods).array().cos()).sum();

      Ctilde(l, m) = 2.0*aux_c + phi_tildes_red(0);
      if (l!=m) Ctilde(m,l) = 2.0*aux_c + phi_tildes_red(0);
    }
  }
  Ctilde.diagonal() = ArrayXd::Constant(means_trans.rows(), 2.*phi_tildes_red.sum() - phi_tildes_red(0));

  return Ctilde;
}

MatrixXd DeterminantalPP::compute_Ctilde_prop(const MatrixXd& means){

  MatrixXd means_trans = ((A * means.transpose()).colwise() + b).transpose();
  MatrixXd Ctilde(means_trans.rows(), means_trans.rows());

  for (int l = 0; l < means_trans.rows()-1; l++) {
    for (int m = l+1; m < means_trans.rows(); m++) {
      //double aux = 0.0;
      //RowVectorXd vec(means_trans.row(l)-means_trans.row(m));
      VectorXd vec_c(means_trans.row(l)-means_trans.row(m));

      //int nthreads;
      //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
      /*for (int kind = 1; kind < Kappas_red.rows(); kind++) {
        //nthreads = omp_get_num_threads();
        //printf("Number of threads = %d\n", nthreads);
        double dotprod = Kappas_red.row(kind).dot(vec);
        aux += phi_tildes_tmp_red[kind] * std::cos(2. * stan::math::pi() * dotprod);
      }*/
      VectorXd dotprods = Kappas_red.bottomRows(Kappas_red.rows()-1) * vec_c;
      double aux_c = (phi_tildes_tmp_red.bottomRows(Kappas_red.rows()-1).array() * (2. * stan::math::pi() * dotprods).array().cos()).sum();

      Ctilde(l, m) = 2.0*aux_c + phi_tildes_tmp_red(0);
      if (l!=m) Ctilde(m,l) = 2.0*aux_c + phi_tildes_tmp_red(0);
    }
  }
  Ctilde.diagonal() = ArrayXd::Constant(means_trans.rows(), 2.*phi_tildes_tmp_red.sum() - phi_tildes_tmp_red(0));

  return Ctilde;
}


void DeterminantalPP::sample_nonalloc_fullcond(MatrixXd *non_active, const MatrixXd &active,
                                 double psi_u, MatrixXd& Ctilde) {
  int npoints = non_active->rows();
  double c_star_na = c_star_red * psi_u;
  birth_prob = std::log(c_star_na) - std::log(c_star_na + npoints);
  double rsecond = uniform_rng(0, 1, Rng::Instance().get());
  birth_arate = -1;
  if (std::log(rsecond) < birth_prob) {
    // BIRTH MOVE
    VectorXd xi = phi_star_rng();
    //std::cout<<"Birth"<<std::endl;
    // compute prob of acceptance of the new birth
    MatrixXd means(active.rows() + npoints, dim);
  //  std::cout<<"defined aux"<<std::endl;
    means << active, *non_active;
    //std::cout<<"filled aux: "<<aux<<std::endl;
  //  std::cout<<"rows= "<<aux.rows()<<std::endl;
  //  std::cout<<"cols= "<<aux.cols()<<std::endl;
  MatrixXd means_trans = ((A*means.transpose()).colwise() + b).transpose();
  VectorXd xi_trans = A * xi + b;
  int n = Ctilde.rows();

  // compute the row/column to be added to Ctilde
  VectorXd col_new = VectorXd::Zero(n);

  for (int l = 0; l < n; l++) {
      //RowVectorXd vec(means_trans.row(l)-xi_trans.transpose());
      VectorXd vec_c(means_trans.row(l)-xi_trans.transpose());

      //int nthreads;
      //#pragma omp parallel for default(none) firstprivate(Kappas,vec, phi_tildes_p) reduction(+:aux)
      /*for (int kind = 1; kind < Kappas_red.rows(); kind++) {
        //nthreads = omp_get_num_threads();
        //printf("Number of threads = %d\n", nthreads);
        double dotprod = Kappas_red.row(kind).dot(vec);
        col_new(l) += 2. * phi_tildes_red[kind] * std::cos(2. * stan::math::pi() * dotprod);
      }*/
      VectorXd dotprods = Kappas_red.bottomRows(Kappas_red.rows()-1) * vec_c;
      col_new(l) = 2.0 * (phi_tildes_red.bottomRows(Kappas_red.rows()-1).array() * (2. * stan::math::pi() * dotprods).array().cos()).sum();

  }
  col_new.array() += phi_tildes_red(0);

  MatrixXd Ctilde_xi = MatrixXd::Zero( n+1, n+1);
  Ctilde_xi.topLeftCorner(n,n) = Ctilde;
  Ctilde_xi.block(0,n,n,1) = col_new;
  Ctilde_xi.block(n,0,1,n) = col_new.transpose();
  Ctilde_xi(n,n) = Ctilde_xi(0,0);

    double pap = papangelou(Ctilde, Ctilde_xi, true);
    //std::cout<<"done papan"<<std::endl;
    birth_arate = pap - phi_star_dens(xi);
    double rthird = uniform_rng(0, 1, Rng::Instance().get());
    if (std::log(rthird) < birth_arate) {
      //std::cout<<"Accepted birth"<<std::endl;
      non_active->conservativeResize(npoints + 1, dim);
      non_active->row(npoints) = xi;
      //Ctilde.resize(n+1,n+1);
      Ctilde = Ctilde_xi;
    }
    //else std::cout<<"Rejected birth"<<std::endl;
  } else {
    // Death Move
    if (npoints == 0) return;
    //std::cout<<"Death"<<std::endl;
    VectorXd probas = VectorXd::Ones(npoints) / npoints;
    int ind = categorical_rng(probas, Rng::Instance().get()) - 1;

    delete_row(non_active, ind);
    delete_row(&Ctilde, ind+active.rows());
    delete_column(&Ctilde, ind+active.rows());
  }

  return;
}
