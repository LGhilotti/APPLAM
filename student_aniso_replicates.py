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
def multivariate_t_rvs(m, S, df=np.inf, n=1, seed = 1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal


def generate_etas_gaussian(mus, deltas_cov, cluster_alloc, seed):
    np.random.seed(seed)
    out = np.vstack([[mvn.rvs(mean = mus[i,:], cov = deltas_cov) for i in cluster_alloc]])
    return out
    
def generate_etas_student(mus, deltas_cov, cluster_alloc, seed):
     np.random.seed(seed)
     #out = np.vstack([[mvn.rvs(mean = mus[i,:], cov = deltas_cov) for i in cluster_alloc]]),
     out = np.vstack([multivariate_t_rvs(m = mus[i,:], S = deltas_cov, df = 3, seed = seed)[0] for i in cluster_alloc])
     return out

def generate_data(Lambda, etas, sigma_bar_cov, seed):
    np.random.seed(seed)
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
DEFAULT_PARAMS_FILE_ANISO = "data/Student_latent_data/resources/sampler_params_aniso.asciipb"
DEFAULT_PARAMS_FILE_ISO = "data/Student_latent_data/resources/sampler_params_iso.asciipb"
#SPECIFIC_PARAMS_FILE = "data/Gaussian_data/resources/comp_pars_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =0
nburn=0
niter = 2000
thin= 1
log_ev=100

ndatasets = 1
n_reruns = 1

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_values", nargs="+", default=["2"])
    parser.add_argument("--d_values", nargs="+", default=["2"])
    parser.add_argument("--m_values", nargs="+", default=["4"])
    parser.add_argument("--n_by_clus", nargs="+", default=["20"])
    args = parser.parse_args()

    p_s = list(map(int, args.p_values))
    d_s = list(map(int, args.d_values))
    M_s = list(map(int, args.m_values))
    n_percluster_s = list(map(int, args.n_by_clus))


    
    # Outer cycle for reading the different datasets and perform the estimation
    for p,dtrue,M,npc in product(p_s, d_s, M_s, n_percluster_s):

        list_performance = list()
        
        for i in range(ndatasets) :

          
          #######################################
          ### READ DATA AND PRE-PROCESSING ######
          #######################################
  
          # read the dataset
          #with open("data/Gaussian_data/datasets/gauss_p_{0}_d_{1}_M_{2}_npc_{3}_data.csv".format(p,dtrue,M,npc), newline='') as my_csv:
          #    data = pd.read_csv(my_csv, sep=',', header=None).values
          
          dist=2
  
          sigma_bar_prec = np.repeat(1000, p)
          sigma_bar_cov = 1/sigma_bar_prec
          
          #lamb = create_lambda(p,dtrue)
          lamb = np.eye(2)
          delta_cov = np.diag(np.array([0.2,0.001]))
          
          #mus = create_mus(dtrue,M,dist)
          mus = np.zeros((M,dtrue))
          mus[0,:] = np.array([3, -1])
          mus[1,:] = np.array([-3, 1])
          mus[2,:] = np.array([-3, -1])
          mus[3,:] = np.array([3, 1])
          
          
          seed = 1222432
          
          cluster_alloc = create_cluster_alloc(npc,M)
          etas = generate_etas_student(mus, delta_cov, cluster_alloc, seed)
          data = generate_data(lamb, etas, sigma_bar_cov, seed)
  
          np.save(file = "data/Student_latent_data/applam/dataset.npy", arr = data)
          
          # scaling of data
          #centering_var=stat.median(np.mean(data,0))
          #print("mean: ", centering_var)
          #scaling_var=stat.median(np.std(data,0))
          #print("std: ", scaling_var)
          #data_scaled=(data-centering_var)/scaling_var
          data_scaled= data
  
          #d = dtrue
          d_s = [2]
  
          for d in d_s:
  
            outpath_d = "data/Student_latent_data/applam/comp_app_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}_out".format(p,d,dtrue,M,npc)
            if not(os.path.exists(outpath_d)):
                os.makedirs(outpath_d)
  
            hyperpar_aniso = Params()
            hyperpar_iso = Params()
            #if os.path.exists(params_file):
             #   print("Using dataset-specific params file for "
            #          "'p'={0}, 'd'={1} 'M'={2}, 'npc'={3}".format(p,d,M,npc))
            #else:
             # print("Using default params file for "
             #         "'p'={0}, 'd'={1} 'M'={2}, 'npc'={3}".format(p,d,M,npc))
            params_file_aniso = DEFAULT_PARAMS_FILE_ANISO
            params_file_iso = DEFAULT_PARAMS_FILE_ISO
  
            with open(params_file_aniso, 'r') as fp:
                text_format.Parse(fp.read(), hyperpar_aniso)
                
            with open(params_file_iso, 'r') as fp:
                text_format.Parse(fp.read(), hyperpar_iso)
  
            
            # ranges
            #ranges = compute_ranges(hyperpar, data_scaled, d)
            ranges = np.array([np.full(d,-6.),np.full(d,6.)])
            
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
  
                
                hyperpar_aniso.dpp.c = c
                hyperpar_aniso.dpp.n = n
                hyperpar_aniso.dpp.s = s
      
                #hyperpar_aniso.wishart.nu = hyperpar_aniso.wishart.nu + d
                
                
                hyperpar_iso.dpp.c = c
                hyperpar_iso.dpp.n = n
                hyperpar_iso.dpp.s = s
      
                #hyperpar_iso.wishart.nu = hyperpar_iso.wishart.nu + d
            
                for j in range(n_reruns):
                
                  #################################################
                  ######## MCMC SAMPLER - APPLAM #############
                  #################################################
    
                  # Build the sampler
                  sampler_aniso = ConditionalMCMC(hyperpar = hyperpar_aniso)
    
                  # Run the algorithm
                  sampler_aniso.run(ntrick, nburn, niter, thin, data_scaled, d, ranges, log_every = log_ev, fix_sigma = "TRUE")
    
    
                  # Save results in the following path
                  #base_outpath_rho = os.path.join(outpath_d, "rho_{0}_out".format(rho)) + "_aniso_{0}"
                  #i = 0
                  #while os.path.exists(base_outpath_rho.format(i)):
                  #    i = i+1
                  #outpath = base_outpath_rho.format(i)
                  #os.makedirs(outpath)
    
                  # Save the serialized chain produced by the sampler
                  #sampler_aniso.serialize_chains(os.path.join(outpath, "chains_aniso.recordio"))
    
                  # Save the parameters
                  #with open(os.path.join(outpath, "params_aniso.asciipb"), 'w') as fp:
                  #    fp.write(text_format.MessageToString(hyperpar))
    
                  # Some plots
                  chain_aniso = sampler_aniso.chains
    
                  n_cluster_chain_aniso = np.array([x.ma for x in chain_aniso])
    
                  n_nonall_chain_aniso = np.array([x.mna for x in chain_aniso])
    
    
                  # Mixing of the lambda parameters
                  fig = plt.figure()
                  first_diag_lambda_chain_aniso = np.array([to_numpy(x.lamb_block.lamb)[0,0] for x in chain_aniso])
                  plt.plot(first_diag_lambda_chain_aniso,color='red')
                  second_diag_lambda_chain_aniso = np.array([to_numpy(x.lamb_block.lamb)[1,1] for x in chain_aniso])
                  plt.plot(second_diag_lambda_chain_aniso,color='blue')
                  ud_diag_lambda_chain_aniso = np.array([to_numpy(x.lamb_block.lamb)[0,1] for x in chain_aniso])
                  plt.plot(ud_diag_lambda_chain_aniso,color='green')
                  ld_diag_lambda_chain_aniso = np.array([to_numpy(x.lamb_block.lamb)[1,0] for x in chain_aniso])
                  plt.plot(ld_diag_lambda_chain_aniso,color='yellow')
                  plt.title("lambda_chain - APPLAM")
                  plt.savefig(os.path.join(outpath_d, "lambda_chain_aniso.pdf") )
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
                  #np.savetxt(os.path.join(outpath, "best_clus_aniso.txt"), best_clus_aniso)
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
                  sampler_iso = ConditionalMCMC_isotropic(hyperpar = hyperpar_iso)
    
                  # Run the algorithm
                  sampler_iso.run(ntrick, nburn, niter, thin, data_scaled, d, ranges, log_every = log_ev, fix_sigma = "TRUE")
    
    
                  # Save results in the following path
                  #base_outpath_rho = os.path.join(outpath_d, "rho_{0}_out".format(rho)) + "_iso_{0}"
                  #i = 0
                  #while os.path.exists(base_outpath_rho.format(i)):
                  #    i = i+1
                  #outpath = base_outpath_rho.format(i)
                  #os.makedirs(outpath)
    
                  # Save the serialized chain produced by the sampler
                  #sampler_iso.serialize_chains(os.path.join(outpath, "chains_iso.recordio"))
    
                  # Save the parameters
                  #with open(os.path.join(outpath, "params_iso.asciipb"), 'w') as fp:
                  #    fp.write(text_format.MessageToString(hyperpar))
    
                  # Some plots
                  chain_iso = sampler_iso.chains
    
                  n_cluster_chain_iso = np.array([x.ma for x in chain_iso])
    
    
                  n_nonall_chain_iso = np.array([x.mna for x in chain_iso])
    
                  
                  # Mixing of the lambda parameters
                  fig = plt.figure()
                  first_diag_lambda_chain_iso = np.array([to_numpy(x.lamb_block.lamb)[0,0] for x in chain_iso])
                  plt.plot(first_diag_lambda_chain_iso,color='red')
                  second_diag_lambda_chain_iso = np.array([to_numpy(x.lamb_block.lamb)[1,1] for x in chain_iso])
                  plt.plot(second_diag_lambda_chain_iso,color='blue')
                  ud_diag_lambda_chain_iso = np.array([to_numpy(x.lamb_block.lamb)[0,1] for x in chain_iso])
                  plt.plot(ud_diag_lambda_chain_iso,color='green')
                  ld_diag_lambda_chain_iso = np.array([to_numpy(x.lamb_block.lamb)[1,0] for x in chain_iso])
                  plt.plot(ld_diag_lambda_chain_iso,color='yellow')
                  plt.title("lambda_chain - ISO")
                  plt.savefig(os.path.join(outpath_d, "lambda_chain_iso.pdf") )
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
                  #np.savetxt(os.path.join(outpath, "best_clus_iso.txt"), best_clus_iso)
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
        df_performance.to_csv(os.path.join(outpath_d, "df_performance.csv"))
