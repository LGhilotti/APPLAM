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
DEFAULT_PARAMS_FILE = "data/Student_lik_repeated_data/resources/sampler_params.asciipb"
SPECIFIC_PARAMS_FILE = "data/Student_lik_repeated_data/resources/sampler_params_p{0}_d{1}_m{2}.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =1000
nburn=2000
niter = 5000
thin= 2
log_ev=50

n_reruns = 1


dist=5



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_values", nargs="+", default=["400"])
    parser.add_argument("--d_values", nargs="+", default=["5"])
    parser.add_argument("--m_values", nargs="+", default=["4"])
    parser.add_argument("--n_by_clus", nargs="+", default=["50"])
    args = parser.parse_args()

    p_s = list(map(int, args.p_values))
    d_s = list(map(int, args.d_values))
    M_s = list(map(int, args.m_values))
    n_percluster_s = list(map(int, args.n_by_clus))


    list_performance_applam = list()
    list_performance_lamb = list()

    # Outer cycle for reading the different datasets and perform the estimation
    for p,d,M,npc in product(p_s, d_s, M_s, n_percluster_s):   


      #######################################
      ### READ DATA AND PRE-PROCESSING ######
      #######################################
      
      seed = 123456
      
      ranges = np.array([np.full(d,-10.),np.full(d,10.)])
      
      sigma_bar_prec = np.repeat(1, p)
      sigma_bar_cov = 1/sigma_bar_prec
      lamb = create_lambda(p,d)     
      #lamb = lamb/np.max(np.sum(lamb, axis=0))     
              
      delta_cov = np.eye(d)
      mus = create_mus(d,M,dist)
      cluster_alloc = create_cluster_alloc(npc,M)
      true_clus = np.repeat(range(M),npc)
      
      outpath_d = "data/Student_lik_repeated_data/applam/app_p_{0}_dtrue_{1}_M_{2}_npc_{3}_out".format(p,d,M,npc)
      if not(os.path.exists(outpath_d)):
          os.makedirs(outpath_d)
          
      for j in range(n_reruns):
      
        seed = seed +1
        
        #lamb = ortho_group.rvs(dim = p, random_state = seed)
        #lamb = 5*lamb[0:p,0:d]
        #print("lamb.shape= ", lamb.shape)
        
        etas = generate_etas_gaussian(mus, delta_cov, cluster_alloc, seed)
        print("eta_first: ", etas[0,])
        print("eta_second: ", etas[1,])
        print("eta_last: ", etas[(npc*M)-1,])
        data = generate_data_student(lamb, etas, sigma_bar_cov, seed)
        print("dim of data: ", data.shape)
        #data = generate_data_gaussian(lamb, etas, sigma_bar_cov, seed)
        print("data_first: ", data[0,0:10])
        print("data_second: ", data[1,0:10])
        print("data_last: ", data[(npc*M)-1,0:10])
        
        # scaling of data
        centering_var=stat.median(np.mean(data,0))
        scaling_var=stat.median(np.std(data,0))
        data_scaled=(data-centering_var)/scaling_var
        print("centering_var: ",centering_var)
        print("scaling_var: ", scaling_var)
        print("datascaled_first: ", data_scaled[0,0:10])
        print("datascaled_second: ", data_scaled[1,0:10])
        print("datascaled_last: ", data_scaled[(npc*M)-1,0:10])
        
                
        #####################
        ### LAMB ############
        #####################
        conc_dir = 1
        nr = np.shape(data_scaled)[0]
        nc = np.shape(data_scaled)[1]
        y_r = robjects.r.matrix(data_scaled, nrow=nr, ncol=nc)
        res_lamb = np.array(DL_mixture_r(y = y_r, d_lamb = d, nrun_lamb = 10**5, burn_lamb = 10**4,  thin_lamb = 5, conc_dir = conc_dir))
        
        res_lamb_df = pd.DataFrame(res_lamb.T)
        nclus_lamb_chain = res_lamb_df.nunique()
        
        print("nclus_chain: ",nclus_lamb_chain)
        
        
        vals_, counts_ = np.unique(nclus_lamb_chain, return_counts=True)
        post_mode_nclus_lamb = vals_[np.argwhere(counts_ == np.max(counts_))][0,0]
        print("mode: ", post_mode_nclus_lamb)
        
        post_avg_nclus_lamb = np.mean(nclus_lamb_chain)
        print("mean: ", post_avg_nclus_lamb)
        
        
        # Best clustering estimate according to Binder's loss
        best_clus_lamb = cluster_estimate(np.array(res_lamb))
        print("best clus: ",best_clus_lamb)
            
        # ARI of the best clustering
        ari_best_clus_lamb = adjusted_rand_score(true_clus, best_clus_lamb)
        # ARIs of the clusterings along the iterations
        aris_chain_lamb = np.apply_along_axis(lambda x,y:adjusted_rand_score(y,x), 1, res_lamb, true_clus)
        mean_aris_lamb, sigma_aris_lamb = np.mean(aris_chain_lamb), np.std(aris_chain_lamb)
        # CI of the previous ARIs
        CI_aris_lamb = norm.interval(0.95, loc=mean_aris_lamb, scale=sigma_aris_lamb/sqrt(len(aris_chain_lamb)))
        
        list_performance_lamb.append(["Lamb", p,d,M,npc, conc_dir, post_mode_nclus_lamb,
                                post_avg_nclus_lamb, ari_best_clus_lamb, CI_aris_lamb])
        

    df_performance_lamb = pd.DataFrame(list_performance_lamb, columns=('model', 'p','d','M','npc','conc_dir',
                                        'mode_nclus', 'avg_nclus', 'ari_best_clus', 'CI_aris'))
    df_performance_lamb.to_csv(os.path.join(outpath_d, "df_performance_lamb.csv"))
  
    