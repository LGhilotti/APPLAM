#ifndef _GIG_HPP
#define _GIG_HPP

#include <cmath>
#include <cfloat>
#include <stan/math/prim.hpp>
#include "rng.hpp"
using namespace stan::math;
using namespace std;


namespace GIG {

  double _gig_mode(double lambda, double omega);

  void _rgig_ROU_shift_alt (double& res, double lambda, double lambda_old, double omega, double alpha);

  void _rgig_newapproach1 (double& res, double lambda, double lambda_old, double omega, double alpha);

  void _rgig_ROU_noshift (double& res, double lambda, double lambda_old, double omega, double alpha);

  #define ZTOL (DBL_EPSILON*10.0)

  double rgig(double lambda, double chi, double psi);

}
#endif
