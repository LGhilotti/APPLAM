#ifndef DETERMINANTAL_PP_HPP
#define DETERMINANTAL_PP_HPP

//#include <stdexcept>
#include <omp.h>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>


#include "../rng.hpp"
#include "../utils.hpp"

using namespace Eigen;
using namespace stan::math;

class DeterminantalPP {
protected:
    MatrixXd ranges; //hyper-Rectangle R
    VectorXd diff_range;
    double vol_range; // |R|
    int dim; // dimension of space PP

    int N; // truncation of the eigendecomposition

    // parameters in the Gaussian spectral density
    double c; // >0 , it sets rho_max
    double s; // in (0,1), s.t. rho= s * rho_max

    // c_star is the integral of psi^* (in b&D algo), which is C_tilde(0)=sum(phis_tilde)
    //double c_star; // c* of X (DPP): changes when Lambda changes!

    // eigendecomposition params: change when Lambda changes!
    /*VectorXd phis;
    VectorXd phi_tildes;
    double Ds;*/

    //MatrixXd Kappas; // grid for approximating summation over Z^dim
    MatrixXd Kappas_red; // grid for approximating summation over Z^dim
    VectorXd phis_red;
    VectorXd phi_tildes_red;
    VectorXd phis_tmp_red;
    VectorXd phi_tildes_tmp_red;
    double Ds_red;
    double Ds_tmp_red;
    double c_star_red;
    double c_star_tmp_red;
    // Affine transformation to the unit square
    MatrixXd A;
    VectorXd b;

    // FOR DEBUGGING
    double birth_prob, birth_arate;

    // FOR PROPOSAL
     const MatrixXd * Lambda; // const pointer to the Lambda matrix
/*
    double c_star_tmp; // c* of X (DPP) wrt the last passed Lambda

    // eigendecomposition params for last passed Lambda
    VectorXd phis_tmp;
    VectorXd phi_tildes_tmp;
    double Ds_tmp;*/

public:

    DeterminantalPP(const MatrixXd& ranges, int N, double c, double s);

    ~DeterminantalPP() {}

    // manages the decomposition in Ds, phi,... for the dpp (using or not lambda)
     // set the pointer to Lambda and performs the initial decomposition
    void set_decomposition(const MatrixXd * lambda);

    // modifies the passed Ds, phis, phi_tildes, c_star according to the dpp defined with lambda
    //void compute_eigen_and_cstar(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda);
    void compute_eigen_and_cstar_red(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda);

    // manages decomposition wrt the passed lambda.
    // it takes the proposed Lambda and performs decomposition, storing it in "tmp" variables
    void decompose_proposal(const MatrixXd& lambda);

    void update_decomposition_from_proposal();

    void compute_Kappas(); // compute just once the grid for summation over Z^dim

    // computes (log default) density in x of cond process wrt the proposed matrix Lambda; USEFUL FOR MultiDpp.
    // For UniDpp it just calls dens_cond (because does not depend on Lambda), for MultiDpp it uses "tmp" variables.
    double dens_cond_in_proposal(const MatrixXd& x, double lndetCtil_prop, bool log=true);

    // computes (log default) density in x of cond process wrt the current decomposition (expressed by the Father variables Ds, phis,...)
    double dens_cond(const MatrixXd& x, double lndetCtil, bool log=true);

    // PRIVATE: final method that actually computes the log-density of dpp given parameters of decomposition
    double ln_dens_process(const MatrixXd& x, double lndetCtil_p, double Ds_p, double c_star_p);

    double papangelou(const MatrixXd& Ctilde, const MatrixXd &Ctilde_xi, bool log = true);

    // sample npoints uniformly in the region defined by range
    MatrixXd sample_uniform(int npoints);

    // Sample from b_bar'(csi)=phi_star(csi)/c_star (used in birth process)
    VectorXd phi_star_rng();

    // Gives (log) of phi_star(xi)
    double phi_star_dens(VectorXd xi, bool log = true);

    MatrixXd compute_Ctilde(const MatrixXd& means);
    MatrixXd compute_Ctilde_prop(const MatrixXd& means);

    // Sample B&D from posterior of mu_na
    void sample_nonalloc_fullcond(
         MatrixXd *non_active, const MatrixXd &active, double psi_u, MatrixXd& Ctilde);


    // GETTERS

    MatrixXd get_ranges() const {return ranges;}

    double get_vol_range() const {return vol_range;}

    int get_dim() const {return dim;}

    double get_c() const {return c;}

    double get_s() const {return s;}

    double get_N() const {return N;}

/*
    double get_cstar() const { return c_star; }

    const MatrixXd& get_kappas() const {return Kappas;}

    const VectorXd& get_phi_tildes() const {return phi_tildes;}

    const VectorXd& get_phis() const {return phis;}

    double get_Ds() const {return Ds;}
*/
    // reduced getters
    double get_cstar_red() const { return c_star_red; }

    const MatrixXd& get_kappas_red() const {return Kappas_red;}

    const VectorXd& get_phi_tildes_red() const {return phi_tildes_red;}

    const VectorXd& get_phis_red() const {return phis_red;}

    double get_Ds_red() const {return Ds_red;}

    const VectorXd& get_phi_tildes_tmp_red() const {return phi_tildes_tmp_red;}

    const VectorXd& get_phis_tmp_red() const {return phis_tmp_red;}

    double get_Ds_tmp_red() const {return Ds_tmp_red;}

    const MatrixXd& get_A() const {return A;}

    const VectorXd& get_b() const {return b;}

};

#endif
