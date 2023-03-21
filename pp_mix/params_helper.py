import logging

import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances

import pp_mix.protos.py.params_pb2 as params_pb2
from pp_mix.protos.py.params_pb2 import Params
from pp_mix.utils import truncated_norm_rng

def check_params(params, data, d):

    if params.WhichOneof("prec_params") not in \
            ("fixed_multi_prec", "wishart"):
        raise ValueError(
            "Found {0} as precision parameter, expected one of: {1}".format(
                params.WhichOneof("prec_params"),
                "[{0}]".format(", ".join(("fixed_multi_prec", "wishart")))
            ))


    if params.WhichOneof("prec_params") == "wishart":
        if params.wishart.nu <= d - 1:
            raise ValueError(
                """Parameter wishart.nu sould be strictly greater than {0} - 1,
                 found wishart.nu={1} instead""".format(
                     d, params.wishart.nu))

        if params.wishart.identity is False:
            raise ValueError(
            "Only 'True' is supported for parametr wishart.identity")

        if params.wishart.sigma <= 0:
            raise ValueError(
            "Parameter wishart.sigma should be grater than 0, "
            "found wishart.wigma={0} instead".format(params.wishart.sigma))

    if params.dpp.c <= 0 or params.dpp.s < 0 or params.dpp.s > 1:
        raise ValueError(
            "Parameter dpp are not admissible.")

    if params.a <= 0 :
        raise ValueError(
            "Parameter a (for Dirichlet) should be grater than 0, "
            "found a={0} instead".format(params.a))

    if params.alphajump <= 0 or params.betajump <= 0 :
        raise ValueError(
            "Gamma parameters for jumps should be greater than 0, "
            "found alphajump={0}, betajump={1}".format(params.alphajump,params.betajump))

    if params.agamma <= 0 or params.bgamma <= 0 :
        raise ValueError(
            "Gamma parameters for Sigma precision should be greater than 0, "
            "found agamma={0}, bgamma={1}".format(params.agamma,params.bgamma))

    if (params.HasField("mh_sigma_means") and params.mh_sigma_means <=0) or (params.HasField("mala_step_means") and params.mala_step_means <=0) :
        raise ValueError(
            "Parameter for allocated means update (MH or Mala) should be grater than 0")


    if (params.HasField("mh_sigma_lambda") and params.mh_sigma_lambda <=0) or (params.HasField("mala_step_lambda") and params.mala_step_lambda <=0) :
        raise ValueError(
            "Parameter for Lambda update (MH or Mala) should be greater than 0")

def compute_ranges(params, data, d):

    p = data.shape[1]
    max_latent = 0
    n_samples = 100

    tau_draws = np.random.gamma(p * d * params.a, 2, size=(n_samples,))
    Psi_draws = np.random.exponential(2.0 , size = (n_samples, p*d))
    Phi_draws = np.random.dirichlet(np.full(p*d, params.a), size=(n_samples,))
    norm_draws = np.random.normal(size=(n_samples, p * d))
    Lambda_draws = Phi_draws * np.sqrt(Psi_draws) * norm_draws
    Lambda_draws = Lambda_draws * tau_draws[:, np.newaxis]
    Lambda = Lambda_draws.reshape((n_samples, p, d))
    lat_fact = np.stack([np.linalg.solve(
        np.dot(Lambda[i, :, :].T, Lambda[i, :, :]),
        np.dot(Lambda[i, :, :].T, data.T)) for i in range(n_samples)])
    max_latent = np.max(np.abs(lat_fact))
    print("max_latent: {0:.4f}".format(max_latent))

    #for i in range(100) :
    #    tau = np.random.gamma(p * d * params.a, 2)
    #    Psi = np.random.exponential(2.0 , size = p*d)
    #    Phi = np.random.dirichlet(np.full(p*d, params.a))
    #    Lambda = (tau * Phi * np.sqrt(Psi) * np.random.normal(size=p*d)).reshape((p,d))

    #   lat_fact = np.linalg.solve(np.dot(Lambda.T,Lambda), np.dot(Lambda.T, data.T))
    #    max_latent = np.max([np.max(np.abs(lat_fact)),max_latent])


    return 10 * np.array([np.full(d,-max_latent),np.full(d,max_latent)])

def compute_ranges_binary(params, binary_data, d):

    p = binary_data.shape[1]
    n = binary_data.shape[0]
    max_latent = 0
    n_samples = 100

    tau_draws = np.random.gamma(p * d * params.a, 2, size=(n_samples,))
    Psi_draws = np.random.exponential(2.0 , size = (n_samples, p*d))
    Phi_draws = np.random.dirichlet(np.full(p*d, params.a), size=(n_samples,))
    norm_draws = np.random.normal(size=(n_samples, p * d))
    Lambda_draws = Phi_draws * np.sqrt(Psi_draws) * norm_draws
    Lambda_draws = Lambda_draws * tau_draws[:, np.newaxis]
    Lambda = Lambda_draws.reshape((n_samples, p, d))

    sigmas_bar = np.random.gamma(params.agamma, params.bgamma, size=(n_samples,p))
    vars = np.stack( [ (Lambda[i,:,:].T * Lambda[i,:,:]).diagonal() + 1/sigmas_bar[i,:] for i in range(n_samples)])

    zetas = truncated_norm_rng(vars, binary_data, size=(n_samples, n, p ))

    lat_fact = np.stack([np.linalg.solve(
        np.dot(Lambda[i, :, :].T, Lambda[i, :, :]),
        np.dot(Lambda[i, :, :].T, zetas[i,:,:].T)) for i in range(n_samples)])
    max_latent = np.max(np.abs(lat_fact))
    print("max_latent: {0:.4f}".format(max_latent))

    #for i in range(100) :
    #    tau = np.random.gamma(p * d * params.a, 2)
    #    Psi = np.random.exponential(2.0 , size = p*d)
    #    Phi = np.random.dirichlet(np.full(p*d, params.a))
    #    Lambda = (tau * Phi * np.sqrt(Psi) * np.random.normal(size=p*d)).reshape((p,d))

    #   lat_fact = np.linalg.solve(np.dot(Lambda.T,Lambda), np.dot(Lambda.T, data.T))
    #    max_latent = np.max([np.max(np.abs(lat_fact)),max_latent])


    return 10 * np.array([np.full(d,-max_latent),np.full(d,max_latent)])


def check_ranges(ranges,d):

    if ranges.shape[1] != d :
        raise ValueError(
            "Ranges columns does not match factor dimension, "
            "found ranges.shape[1]={0}, dimf={1}".format(ranges.shape[1],d))
    if ranges.shape[0] != 2:
        raise ValueError(
            "Ranges should have 2 rows, incorrect number of rows, "
            "found ranges.shape[0]={0}".format(ranges.shape[0]))
