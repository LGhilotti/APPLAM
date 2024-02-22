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
from scipy.stats import ortho_group
from scipy.interpolate import griddata
from sklearn.metrics import adjusted_rand_score
from math import sqrt
import math
from itertools import product
from sklearn.decomposition import TruncatedSVD
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import sys
sys.path.append('.')
sys.path.append('./pp_mix')

from pp_mix.interface import ConditionalMCMC, ConditionalMCMC_isotropic, cluster_estimate
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params


def multivariate_t_rvs(m, S, df=np.inf, n=1, seed = 1234):
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
    np.random.seed(seed)
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

def generate_data_gaussian(Lambda, etas, sigma_bar_cov, seed):
    np.random.seed(seed)
    means = np.matmul(Lambda,etas.T)
    sigma_bar_cov_mat = np.diag(sigma_bar_cov)
    out = np.vstack([mvn.rvs(mean = means[:,i], cov = sigma_bar_cov_mat) for i in range(etas.shape[0])])
    return out
    
def generate_data_student(Lambda, etas, sigma_bar_cov, seed):
    means = np.matmul(Lambda,etas.T)
    sigma_bar_cov_mat = np.diag(sigma_bar_cov)
    out = np.vstack([multivariate_t_rvs(m = means[:,i], S = sigma_bar_cov_mat, df = 3, seed = seed)[0] for i in range(etas.shape[0])])
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


###############################
##### R functions for Lamb ####
###############################

# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('lamb.R')

# Loading the function we have defined in R.
DL_mixture_r = robjects.globalenv['DL_mixture_function']

##############################################
# COMMON QUANTITIES TO ALL RUNS OF ALGORITHM #
##############################################

# Set hyperparameters (agreeing with Chandra)
DEFAULT_PARAMS_FILE = "data/Student_lik_repeated_data_n1000/resources/sampler_params.asciipb"
SPECIFIC_PARAMS_FILE = "data/Student_lik_repeated_data_n1000/resources/sampler_params_p{0}_d{1}_m{2}.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =500
nburn=2000
niter = 10000



thin= 2
log_ev=50

n_reruns = 1


dist=5



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_values", nargs="+", default=["500"])
    parser.add_argument("--d_values", nargs="+", default=["4"])
    parser.add_argument("--m_values", nargs="+", default=["4"])
    parser.add_argument("--n_by_clus", nargs="+", default=["250"])
    args = parser.parse_args()

    p_s = list(map(int, args.p_values))
    d_s = list(map(int, args.d_values))
    M_s = list(map(int, args.m_values))
    n_percluster_s = list(map(int, args.n_by_clus))

    

    # Outer cycle for reading the different datasets and perform the estimation
    for p,d,M,npc in product(p_s, d_s, M_s, n_percluster_s):   

      list_performance_applam = list()
      list_performance_lamb = list()

      #######################################
      ### READ DATA AND PRE-PROCESSING ######
      #######################################
      
      seed = 12345 
      
      ranges = np.array([np.full(d,-10.),np.full(d,10.)])
      
      sigma_bar_prec = np.repeat(0.1, p)
      sigma_bar_cov = 1/sigma_bar_prec
      #lamb = create_lambda(p,d)
      delta_cov = np.eye(d)
      mus = create_mus(d,M,dist)
      cluster_alloc = create_cluster_alloc(npc,M)
      true_clus = np.repeat(range(M),npc)
      
      outpath_d = "data/Student_lik_repeated_data_n1000/applam/app_p_{0}_dtrue_{1}_M_{2}_npc_{3}_out".format(p,d,M,npc)
      if not(os.path.exists(outpath_d)):
          os.makedirs(outpath_d)
          
      for j in range(n_reruns):
      
        seed = seed +1
        lamb = ortho_group.rvs(dim = p, random_state = seed)
        lamb = lamb[0:p,0:d]
        print("lamb.shape= ", lamb.shape)
        
        etas = generate_etas_gaussian(mus, delta_cov, cluster_alloc, seed)
        data = generate_data_student(lamb, etas, sigma_bar_cov, seed)
        #data = generate_data_gaussian(lamb, etas, sigma_bar_cov, seed)

        # scaling of data
        centering_var=stat.median(np.mean(data,0))
        scaling_var=stat.median(np.std(data,0))
        data_scaled=(data-centering_var)/scaling_var
        
                
        #####################
        ## APPLAM ###########
        #####################
        col_mean =np.mean(data,axis = 0)
        
        data_zero_mean = data - col_mean
        
        U, S, Vh = np.linalg.svd(data_zero_mean, full_matrices=True)
        
        lamb_est = Vh[0:d, ].transpose()

        

        ####################################
        ##### HYPERPARAMETERS ##############
        ####################################
        hyperpar = Params()
        params_file = SPECIFIC_PARAMS_FILE.format(p,d,M)
        if os.path.exists(params_file):
            print("Using dataset-specific params file for "
                  "'p'={0}, 'd'={1} 'M'={2}".format(p,d,M))
        else:
            print("Using default params file for "
                  "'p'={0}, 'd'={1} 'M'={2}".format(p,d,M))
            params_file = DEFAULT_PARAMS_FILE
        with open(params_file, 'r') as fp:
            text_format.Parse(fp.read(), hyperpar)
      
        # Set the expected number of centers a priori
        #square_vol = (ranges[1,0]*2)**d
        #expected_points = [1,5,10]        
        #rho_s = expected_points/square_vol
        
        rho_s = [0.5, 1, 5]

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
            sampler_aniso.run(ntrick, nburn, niter, thin, data, d, lamb_est, ranges, n_init_centers = 4, fix_lambda = "FALSE", fix_sigma = "FALSE", log_every = log_ev)


            # Save results in the following path
            base_outpath_rho = os.path.join(outpath_d, "rho_{0}_out".format(rho)) 
            if not(os.path.exists(base_outpath_rho)):
                 os.makedirs(base_outpath_rho)

            
            # Some plots
            chain_aniso = sampler_aniso.chains
            
            n_cluster_chain_aniso = np.array([x.ma for x in chain_aniso])
          
            n_nonall_chain_aniso = np.array([x.mna for x in chain_aniso])

            # Mixing of tau parameter
            if j == 0 :
              fig = plt.figure()
              tau_chain_aniso = np.array([x.lamb_block.tau for x in chain_aniso])
              plt.plot(tau_chain_aniso)
              plt.title("tau chain - APPLAM")
              plt.savefig(os.path.join(base_outpath_rho, "tau_chain_aniso.pdf"))
              plt.close()

              # Mixing of the sbar parameters
              fig = plt.figure()
              first_sbar_chain_aniso = np.array([to_numpy(x.sigma_bar)[0] for x in chain_aniso])
              plt.plot(first_sbar_chain_aniso,color='red')
              last_sbar_chain_aniso = np.array([to_numpy(x.sigma_bar)[-1] for x in chain_aniso])
              plt.plot(last_sbar_chain_aniso,color='blue')
              plt.title("sbar_chain - APPLAM")
              plt.savefig(os.path.join(base_outpath_rho, "sbar_chain_aniso.pdf"))
              plt.close()
              
              
              # Mixing of the Delta parameters
              fig = plt.figure()
              first_delta_chain_aniso = np.array([to_numpy(x.a_deltas[0])[0,0] for x in chain_aniso])
              plt.plot(first_delta_chain_aniso,color='red')
              last_delta_chain_aniso = np.array([to_numpy(x.a_deltas[0])[-1,-1] for x in chain_aniso])
              plt.plot(last_delta_chain_aniso,color='blue')
              plt.title("a_delta_chain - APPLAM")
              plt.savefig(os.path.join(base_outpath_rho, "a_delta_chain_aniso.pdf"))
              plt.close()

              # Chain of the number of clusters
              fig = plt.figure()
              n_cluster_chain_aniso = np.array([x.ma for x in chain_aniso])
              plt.plot(n_cluster_chain_aniso)
              plt.title("number of clusters chain - APPLAM")
              plt.savefig(os.path.join(base_outpath_rho, "nclus_chain_aniso.pdf"))
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
            
            # ARI of the best clustering
            ari_best_clus_aniso = adjusted_rand_score(true_clus, best_clus_aniso)
            # ARIs of the clusterings along the iterations
            aris_chain_aniso = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain_aniso])
            mean_aris_aniso, sigma_aris_aniso = np.mean(aris_chain_aniso), np.std(aris_chain_aniso)
            # CI of the previous ARIs
            CI_aris_aniso = norm.interval(0.95, loc=mean_aris_aniso, scale=sigma_aris_aniso/sqrt(len(aris_chain_aniso)))


            ###########################################################################
            ####### Save inferred quantities in dataframe ############################
            #########################################################################

            list_performance_applam.append([j, "APPLAM", p,d,M,npc,sampler_aniso.means_ar, sampler_aniso.lambda_ar, rho, post_mode_nclus_aniso,
                                post_avg_nclus_aniso, post_avg_nonall_aniso, ari_best_clus_aniso, CI_aris_aniso])

          
        

      df_performance_applam = pd.DataFrame(list_performance_applam, columns=('n_dataset', 'model', 'p','d','M','npc','means_ar','lambda_ar', 'intensity', 'mode_nclus', 'avg_nclus', 'avg_nonalloc', 'ari_best_clus', 'CI_aris'))
      df_performance_applam.to_csv(os.path.join(outpath_d, "df_performance_applam_p{0}_d{1}.csv".format(p,d)))
      
    