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
from scipy.stats import skewnorm, t
from scipy.stats import norm, mode
from scipy.stats import ortho_group
from scipy.interpolate import griddata
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
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
    
    
##############################################
# COMMON QUANTITIES TO ALL RUNS OF ALGORITHM #
##############################################

# Set hyperparameters (agreeing with Chandra)
DEFAULT_PARAMS_FILE_ANISO = "data/Fixed_lambda_multidim/resources/sampler_params_aniso.asciipb"
DEFAULT_PARAMS_FILE_ISO = "data/Fixed_lambda_multidim/resources/sampler_params_iso.asciipb"
#SPECIFIC_PARAMS_FILE = "data/Gaussian_data/resources/comp_pars_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =5000
nburn= 50000
niter = 5000
thin= 2
log_ev=500

n_reruns = 1

list_performance = list()



  
#######################################
### READ DATA AND PRE-PROCESSING ######
#######################################
p=100
d=dtrue=2
npc=500
M=2

#sigma_bar_prec = np.repeat(0.1, p)
sigma_bar_prec = np.repeat(0.1, p)

sigma_bar_cov = 1/sigma_bar_prec





lamb_lat = np.array([[3,0],[0,1/3]])

#V = np.eye(p)
seed_V = 122243
V = ortho_group.rvs(dim = p, random_state = seed_V)
V = V[0:p,0:d]
print("V.shape= ", V.shape)

lamb = np.matmul(V,lamb_lat)

delta_cov = np.eye(d)


seed = 122243
np.random.seed(seed)

#etas = np.matrix(t.rvs(10, 0, 1, size = npc).reshape(npc,1))
#print(etas)
#etas_1 = np.matrix(t.rvs(10, -1.5, 0.5, size = npc).reshape(npc,1))
#etas_1 = np.random.normal(loc = -1.5, scale = 0.5, size = (npc,1))
etas_1 = np.empty((npc,2))
etas_1[:,0] = np.random.normal(loc = -1.5, scale = 0.5, size = npc)
etas_1[:,1] = np.random.normal(loc = 0, scale = 1.5, size = npc)

#etas_2 = np.matrix(t.rvs(10, 1.5, 0.5, size = npc).reshape(npc,1))
#etas_2 = np.random.normal(loc = 1.5, scale = 0.5, size = (npc,1))
etas_2 = np.empty((npc,2))
etas_2[:,0] = np.random.normal(loc = 1.5, scale = 0.5, size = npc)
etas_2[:,1] = np.random.normal(loc = 0, scale = 1.5, size = npc)

etas = np.concatenate((etas_1,etas_2))
print(etas)

data = np.matmul(lamb,etas.T).T +  np.vstack([mvn.rvs(cov = np.diag(sigma_bar_cov)) for i in range(etas.shape[0])])

np.savetxt('data/Fixed_lambda_multidim/applam/dataset.csv', data, delimiter=',', fmt='%f')

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

  outpath_d = "data/Fixed_lambda_multidim/applam/comp_app_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}_out".format(p,d,dtrue,M,npc)
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
  ranges = compute_ranges(lamb, data_scaled, d)
  #ranges = np.array([np.full(d,-6.),np.full(d,6.)])
  
  ####################################
  ##### HYPERPARAMETERS ##############
  ####################################
  
  
  # Set the expected number of centers a priori
  rho_s = [1]
  
  for rho in rho_s:
  
      # Fix "s", then: rho_max = rho/s
      # It follows: c = rho_max * (2 pi)^{d/2}
      s_aniso = 0.9
      rho_max_aniso = rho/s_aniso
      c_aniso = rho_max_aniso * ((2. * np.pi) ** (float(d)/2))
      print("s_aniso: ",s_aniso)
  
      hyperpar_aniso.dpp.c = c_aniso
      hyperpar_aniso.dpp.n = n
      hyperpar_aniso.dpp.s = s_aniso
  
      #hyperpar_aniso.wishart.nu = hyperpar_aniso.wishart.nu + d
      
      s_iso = s_aniso * abs(lamb_lat[1,1])/abs(lamb_lat[0,0])
      rho_max_iso = rho/s_iso
      c_iso = rho_max_iso * ((2. * np.pi) ** (float(d)/2))
      print("s_iso: ",s_iso)
      
     
      
      hyperpar_iso.dpp.c = c_iso
      hyperpar_iso.dpp.n = n
      hyperpar_iso.dpp.s = s_iso
  
      #hyperpar_iso.wishart.nu = hyperpar_iso.wishart.nu + d
      
  
      for j in range(n_reruns):
      
        #################################################
        ######## MCMC SAMPLER - APPLAM #############
        #################################################
  
        # Build the sampler
        sampler_aniso = ConditionalMCMC(hyperpar = hyperpar_aniso)
  
            
        # Run the algorithm
        sampler_aniso.run(ntrick, nburn, niter, thin, data_scaled, d, lamb, ranges, fix_lambda = "TRUE", fix_sigma = "TRUE", log_every = log_ev)
  
  
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
  
  
        # Mixing of the delta parameters
        #fig = plt.figure()
        #first_all_delta_chain_aniso = np.array([to_numpy(x.a_deltas[0])[0,0] for x in chain_aniso])
        #plt.plot(first_all_delta_chain_aniso,color='red')
        #plt.title("alloc delta - APPLAM")
        #plt.savefig(os.path.join(outpath_d, "alloc_delta_chain_aniso.pdf") )
        #plt.close()
        
        # Mixing of the lambda parameters
        fig = plt.figure()
        first_diag_lambda_chain_aniso = np.array([to_numpy(x.lamb_block.lamb)[0,0] for x in chain_aniso])
        plt.plot(first_diag_lambda_chain_aniso,color='red')
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
        #ari_best_clus_aniso = adjusted_rand_score(true_clus, best_clus_aniso)
        ri_best_clus_aniso = rand_score(true_clus, best_clus_aniso)
        # ARIs of the clusterings along the iterations
        #aris_chain_aniso = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain_aniso])
        #mean_aris_aniso, sigma_aris_aniso = np.mean(aris_chain_aniso), np.std(aris_chain_aniso)
        ris_chain_aniso = np.array([rand_score(true_clus, x) for x in clus_alloc_chain_aniso])
        mean_ris_aniso, sigma_ris_aniso = np.mean(ris_chain_aniso), np.std(ris_chain_aniso)
        # CI of the previous ARIs
        #CI_aris_aniso = norm.interval(0.95, loc=mean_aris_aniso, scale=sigma_aris_aniso/sqrt(len(aris_chain_aniso)))
        CI_ris_aniso = norm.interval(0.95, loc=mean_ris_aniso, scale=sigma_ris_aniso/sqrt(len(ris_chain_aniso)))
  
  
        #Last allocated means
        last_a_means_chain_aniso = np.array([to_numpy(x) for x in chain_aniso[-1].a_means]) 
        print(last_a_means_chain_aniso)
        
        #Last clustering
        last_clustering_aniso = clus_alloc_chain_aniso[-1]
        unique_aniso, counts_aniso = np.unique(last_clustering_aniso, return_counts=True)
        print(np.asarray((unique_aniso, counts_aniso)).T)
  
        #################################################
        ######## MCMC SAMPLER - ISOTROPIC #############
        #################################################
        
        # Build the sampler
        sampler_iso = ConditionalMCMC_isotropic(hyperpar = hyperpar_iso)
  
        # Run the algorithm
        sampler_iso.run(ntrick, nburn, niter, thin, data_scaled, d, lamb, ranges,  fix_lambda = "TRUE", fix_sigma = "TRUE", log_every = log_ev)
  
  
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
        #ari_best_clus_aniso = adjusted_rand_score(true_clus, best_clus_aniso)
        ri_best_clus_iso = rand_score(true_clus, best_clus_iso)
        # ARIs of the clusterings along the iterations
        #aris_chain_aniso = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain_aniso])
        #mean_aris_aniso, sigma_aris_aniso = np.mean(aris_chain_aniso), np.std(aris_chain_aniso)
        ris_chain_iso = np.array([rand_score(true_clus, x) for x in clus_alloc_chain_iso])
        mean_ris_iso, sigma_ris_iso = np.mean(ris_chain_iso), np.std(ris_chain_iso)
        # CI of the previous ARIs
        #CI_aris_aniso = norm.interval(0.95, loc=mean_aris_aniso, scale=sigma_aris_aniso/sqrt(len(aris_chain_aniso)))
        CI_ris_iso = norm.interval(0.95, loc=mean_ris_iso, scale=sigma_ris_iso/sqrt(len(ris_chain_iso)))
  
  
         #Last allocated means
        last_a_means_chain_iso = np.array([to_numpy(x) for x in chain_iso[-1].a_means]) 
        print(last_a_means_chain_iso)
        
        #Last clustering
        last_clustering_iso = clus_alloc_chain_iso[-1]
        unique_iso, counts_iso = np.unique(last_clustering_iso, return_counts=True)
        print(np.asarray((unique_iso, counts_iso)).T)
        
        ###########################################################################
        ####### Save inferred quantities in dataframe ############################
        #########################################################################
  
        list_performance.append(["APPLAM", p,dtrue,d,M,npc,sampler_aniso.means_ar, sampler_aniso.lambda_ar, rho, post_mode_nclus_aniso,
                            post_avg_nclus_aniso, post_avg_nonall_aniso, ri_best_clus_aniso, CI_ris_aniso, ranges[0][0]])
  
        list_performance.append(["Isotropic", p,dtrue,d,M,npc,sampler_iso.means_ar, sampler_iso.lambda_ar, rho, post_mode_nclus_iso,
                            post_avg_nclus_iso, post_avg_nonall_iso, ri_best_clus_iso, CI_ris_iso, ranges[0][0]])
  
  

  
df_performance = pd.DataFrame(list_performance, columns=('model', 'p','dtrue','d','M','npc','means_ar','lambda_ar', 'intensity',
                                    'mode_nclus', 'avg_nclus', 'avg_nonalloc', 'ri_best_clus', 'CI_ris', 'ranges'))
df_performance.to_csv(os.path.join(outpath_d, "df_performance.csv"))
