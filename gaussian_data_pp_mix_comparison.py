# NOTE: TO BE LAUNCHED FROM pp_mix


#############################################
### IMPORT LIBRARIES AND FUNCTIONS ##########
#############################################

import argparse
import numpy as np
import os
import pandas as pd
import statistics as stat

# import pymc3 as pm
import pickle
import matplotlib.pyplot as plt
from google.protobuf import text_format
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm, mode
from scipy.interpolate import griddata
from sklearn.metrics import adjusted_rand_score
from math import sqrt
import math
from itertools import product
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.append('.')
sys.path.append('./pp_mix')

from pp_mix.interface import ConditionalMCMC, ConditionalMCMC_isotropic, cluster_estimate
from pp_mix.params_helper import compute_ranges
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params


##############################################
### FUNCTIONS TO GENERATE DATA ##############
##############################################
def generate_etas(mus, deltas_cov, cluster_alloc):
    np.random.seed(seed=233423)
    out = np.vstack([[mvn.rvs(mean = mus[i,:], cov = deltas_cov) for i in cluster_alloc]])
    return out

def generate_data(Lambda, etas, sigma_bar_cov):
    np.random.seed(seed=233423)
    means = np.matmul(Lambda,etas.T)
    sigma_bar_cov_mat = np.diag(sigma_bar_cov)
    out = np.vstack([mvn.rvs(mean = means[:,i], cov = sigma_bar_cov_mat) for i in range(etas.shape[0])])
    return out

def create_lambda(p,d):
    #if p % d != 0:
      #  raise ValueError("Non compatible dimensions p and d: p={0}, d={1}".format(p,d))

    h = math.ceil(p/d)
    Lambda=np.zeros((p,d))
    for i in range(d-1):
        Lambda[i*h:i*h+h,i] = np.ones(h)

    Lambda[(d-1)*h:,d-1] = np.ones(p-(d-1)*h)
    return Lambda

def create_mus(d,M,dist):
    mus = np.zeros((M,d))
    tot_range = (M-1)*dist
    max_mu = tot_range/2
    for i in range(M):
        mus[i,:] = np.repeat(max_mu-i*dist, d)

    return mus

def create_cluster_alloc(n_pc,M):
    return np.repeat(range(M),n_pc)
    
    
##############################################
# COMMON QUANTITIES TO ALL RUNS OF ALGORITHM #
##############################################

# Set hyperparameters (agreeing with Chandra)
DEFAULT_PARAMS_FILE = "data/Gaussian_data/resources/sampler_params.asciipb"
SPECIFIC_PARAMS_FILE = "data/Gaussian_data/resources/comp_pars_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =1000
nburn=5000
niter = 8000
thin= 5
log_ev=100

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_values", nargs="+", default=["10"])
    parser.add_argument("--d_values", nargs="+", default=["2"])
    parser.add_argument("--m_values", nargs="+", default=["4"])
    parser.add_argument("--n_by_clus", nargs="+", default=["50"])
    args = parser.parse_args()

    p_s = list(map(int, args.p_values))
    d_s = list(map(int, args.d_values))
    M_s = list(map(int, args.m_values))
    n_percluster_s = list(map(int, args.n_by_clus))


    np.random.seed(123456)
    
    # Outer cycle for reading the different datasets and perform the estimation
    for p,dtrue,M,npc in product(p_s, d_s, M_s, n_percluster_s):

        list_performance = list()

        #######################################
        ### READ DATA AND PRE-PROCESSING ######
        #######################################

        # read the dataset
        #with open("data/Gaussian_data/datasets/gauss_p_{0}_d_{1}_M_{2}_npc_{3}_data.csv".format(p,dtrue,M,npc), newline='') as my_csv:
        #    data = pd.read_csv(my_csv, sep=',', header=None).values
        
        dist=2

        sigma_bar_prec = np.repeat(1000, p)
        sigma_bar_cov = 1/sigma_bar_prec
        
        lamb = create_lambda(p,dtrue)
        delta_cov = np.eye(dtrue)*0.1
        
        mus = create_mus(dtrue,M,dist)
        
        cluster_alloc = create_cluster_alloc(npc,M)
        etas = generate_etas(mus, delta_cov, cluster_alloc)
        data = generate_data(lamb, etas, sigma_bar_cov)

        # scaling of data
        centering_var=stat.median(np.mean(data,0))
        print("mean: ", centering_var)
        scaling_var=stat.median(np.std(data,0))
        print("std: ", scaling_var)
        data_scaled=(data-centering_var)/scaling_var
        #data_scaled= data

        #d = dtrue
        d_s = [4]

        for d in d_s:

          outpath_d = "data/Gaussian_data/applam/comp_app_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}_out".format(p,d,dtrue,M,npc)
          if not(os.path.exists(outpath_d)):
              os.makedirs(outpath_d)

          hyperpar = Params()
          params_file = SPECIFIC_PARAMS_FILE.format(p,d,dtrue,M,npc)
          #if os.path.exists(params_file):
           #   print("Using dataset-specific params file for "
          #          "'p'={0}, 'd'={1} 'M'={2}, 'npc'={3}".format(p,d,M,npc))
          #else:
           # print("Using default params file for "
           #         "'p'={0}, 'd'={1} 'M'={2}, 'npc'={3}".format(p,d,M,npc))
          params_file = DEFAULT_PARAMS_FILE

          with open(params_file, 'r') as fp:
              text_format.Parse(fp.read(), hyperpar)

          
          # ranges
          #ranges = compute_ranges(hyperpar, data_scaled, d)
          ranges = np.array([np.full(d,-10.),np.full(d,10.)])
          
          ####################################
          ##### HYPERPARAMETERS ##############
          ####################################


          # Set the expected number of centers a priori
          rho_s = [10]

          for rho in rho_s:

              # Fix "s", then: rho_max = rho/s
              # It follows: c = rho_max * (2 pi)^{d/2}
              s = 0.5
              rho_max = rho/s
              c = rho_max * ((2. * np.pi) ** (float(d)/2))

              
              hyperpar.dpp.c = c
              hyperpar.dpp.n = n
              hyperpar.dpp.s = s
    
              hyperpar.wishart.nu = hyperpar.wishart.nu + d
          
              #################################################
              ######## MCMC SAMPLER - APPLAM #############
              #################################################

             
              # Build the sampler
              sampler_aniso = ConditionalMCMC(hyperpar = hyperpar)

              # Run the algorithm
              sampler_aniso.run(ntrick, nburn, niter, thin, data_scaled, d, ranges, log_every = log_ev)


              # Save results in the following path
              base_outpath_rho = os.path.join(outpath_d, "rho_{0}_out".format(rho)) + "_aniso_{0}"
              i = 0
              while os.path.exists(base_outpath_rho.format(i)):
                  i = i+1
              outpath = base_outpath_rho.format(i)
              os.makedirs(outpath)

              # Save the serialized chain produced by the sampler
              #sampler_aniso.serialize_chains(os.path.join(outpath, "chains_aniso.recordio"))

              # Save the parameters
              with open(os.path.join(outpath, "params_aniso.asciipb"), 'w') as fp:
                  fp.write(text_format.MessageToString(hyperpar))

              # Some plots
              chain_aniso = sampler_aniso.chains

              n_cluster_chain_aniso = np.array([x.ma for x in chain_aniso])

              n_nonall_chain_aniso = np.array([x.mna for x in chain_aniso])

              # Chain of the number of clusters
              fig = plt.figure()
              plt.plot(n_cluster_chain_aniso)
              plt.title("number of clusters chain - APPLAM")
              plt.savefig(os.path.join(outpath, "nclus_chain_aniso.pdf"))
              plt.close()
              
              ##################################################################
              ####### Compute quantities for summarizing performance - APPLAM ###########
              #################################################################

              # Posterior mode of number of clusters
              post_mode_nclus_aniso = mode(n_cluster_chain_aniso)[0][0]
              # Posterior mean of number of clusters
              post_avg_nclus_aniso = n_cluster_chain_aniso.mean()
              # Posterior mean of number of non-allocated components
              post_avg_nonall_aniso =  n_nonall_chain_aniso.mean()
              # Cluster allocations along the iterations
              clus_alloc_chain_aniso = [x.clus_alloc for x in chain_aniso]
              # Best clustering estimate according to Binder's loss
              best_clus_aniso = cluster_estimate(np.array(clus_alloc_chain_aniso))
              # Save best cluster estimate
              np.savetxt(os.path.join(outpath, "best_clus_aniso.txt"), best_clus_aniso)
              # Number of cluster of best clustering
              n_clus_best_clus_aniso = np.size(np.unique(best_clus_aniso))
              true_clus = np.repeat(range(M),npc)
              # ARI of the best clustering
              ari_best_clus_aniso = adjusted_rand_score(true_clus, best_clus_aniso)
              # ARIs of the clusterings along the iterations
              aris_chain_aniso = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain_aniso])
              mean_aris_aniso, sigma_aris_aniso = np.mean(aris_chain_aniso), np.std(aris_chain_aniso)
              # CI of the previous ARIs
              CI_aris_aniso = norm.interval(0.95, loc=mean_aris_aniso, scale=sigma_aris_aniso/sqrt(len(aris_chain_aniso)))


              #################################################
              ######## MCMC SAMPLER - ISOTROPIC #############
              #################################################

              # Build the sampler
              sampler_iso = ConditionalMCMC_isotropic(hyperpar = hyperpar)

              # Run the algorithm
              sampler_iso.run(ntrick, nburn, niter, thin, data_scaled, d, ranges, log_every = log_ev)


              # Save results in the following path
              base_outpath_rho = os.path.join(outpath_d, "rho_{0}_out".format(rho)) + "_iso_{0}"
              i = 0
              while os.path.exists(base_outpath_rho.format(i)):
                  i = i+1
              outpath = base_outpath_rho.format(i)
              os.makedirs(outpath)

              # Save the serialized chain produced by the sampler
              #sampler_iso.serialize_chains(os.path.join(outpath, "chains_iso.recordio"))

              # Save the parameters
              with open(os.path.join(outpath, "params_iso.asciipb"), 'w') as fp:
                  fp.write(text_format.MessageToString(hyperpar))

              # Some plots
              chain_iso = sampler_iso.chains

              n_cluster_chain_iso = np.array([x.ma for x in chain_iso])


              n_nonall_chain_iso = np.array([x.mna for x in chain_iso])

              # Chain of the number of clusters
              fig = plt.figure()
              plt.plot(n_cluster_chain_iso)
              plt.title("number of clusters chain - ISO")
              plt.savefig(os.path.join(outpath, "nclus_chain_iso.pdf"))
              plt.close()
              
              ##################################################################
              ####### Compute quantities for summarizing performance ###########
              #################################################################

              # Posterior mode of number of clusters
              post_mode_nclus_iso = mode(n_cluster_chain_iso)[0][0]
              # Posterior mean of number of clusters
              post_avg_nclus_iso = n_cluster_chain_iso.mean()
              # Posterior mean of number of non-allocated components
              post_avg_nonall_iso =  n_nonall_chain_iso.mean()
              # Cluster allocations along the iterations
              clus_alloc_chain_iso = [x.clus_alloc for x in chain_iso]
              # Best clustering estimate according to Binder's loss
              best_clus_iso = cluster_estimate(np.array(clus_alloc_chain_iso))
              # Save best cluster estimate
              np.savetxt(os.path.join(outpath, "best_clus_iso.txt"), best_clus_iso)
              # Number of cluster of best clustering
              n_clus_best_clus_iso = np.size(np.unique(best_clus_iso))
              true_clus = np.repeat(range(M),npc)
              # ARI of the best clustering
              ari_best_clus_iso = adjusted_rand_score(true_clus, best_clus_iso)
              # ARIs of the clusterings along the iterations
              aris_chain_iso = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain_iso])
              mean_aris_iso, sigma_aris_iso = np.mean(aris_chain_iso), np.std(aris_chain_iso)
              # CI of the previous ARIs
              CI_aris_iso = norm.interval(0.95, loc=mean_aris_iso, scale=sigma_aris_iso/sqrt(len(aris_chain_iso)))


              ###########################################################################
              ####### Save inferred quantities in dataframe ############################
              #########################################################################

              list_performance.append(["APPLAM", p,dtrue,d,M,npc,sampler_aniso.means_ar, sampler_aniso.lambda_ar, rho, post_mode_nclus_aniso,
                                  post_avg_nclus_aniso, post_avg_nonall_aniso, ari_best_clus_aniso, CI_aris_aniso, ranges[0][0]])

              list_performance.append(["Isotropic", p,dtrue,d,M,npc,sampler_iso.means_ar, sampler_iso.lambda_ar, rho, post_mode_nclus_iso,
                                  post_avg_nclus_iso, post_avg_nonall_iso, ari_best_clus_iso, CI_aris_iso, ranges[0][0]])




          df_performance = pd.DataFrame(list_performance, columns=('model', 'p','dtrue','d','M','npc','means_ar','lambda_ar', 'intensity',
                                              'mode_nclus', 'avg_nclus', 'avg_nonalloc', 'ari_best_clus', 'CI_aris', 'ranges'))
          df_performance.to_csv(os.path.join(outpath, "df_performance.csv"))
