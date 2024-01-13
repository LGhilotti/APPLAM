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
DEFAULT_PARAMS_FILE_ANISO = "data/Free_svd_lambda_multidim/resources/sampler_params_aniso.asciipb"
DEFAULT_PARAMS_FILE_ISO = "data/Free_svd_lambda_multidim/resources/sampler_params_iso.asciipb"
#SPECIFIC_PARAMS_FILE = "data/Gaussian_data/resources/comp_pars_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =500
#ntrick =5000
nburn= 500
#nburn= 25000
niter = 500
#niter = 10000

thin= 2
log_ev=500

n_reruns = 3

list_performance = list()



  
#######################################
### READ DATA AND PRE-PROCESSING ######
#######################################
      
p=100
d=dtrue=2
npc=500
M=2

outpath_d = "data/Free_svd_lambda_multidim/applam/comp_app_p_{0}_d_{1}_dtrue_{2}_M_{3}_npc_{4}_out".format(p,d,dtrue,M,npc)

if not(os.path.exists(outpath_d)):
      os.makedirs(outpath_d)
      
      
#sigma_bar_prec = np.repeat(0.1, p)
sigma_bar_prec = np.repeat(0.1, p)

sigma_bar_cov = 1/sigma_bar_prec

delta_cov = np.eye(d)

lamb_lat = np.array([[3,0],[0,1/3]])

#V = np.eye(p)
seed_V = 122243
seed = 122243
np.random.seed(seed)

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
#ranges = compute_ranges(lamb_est, data_scaled, d)
ranges = np.array([np.full(d,-10.),np.full(d,10.)])

for j in range(n_reruns):

  print("Dataset number ", j)

  V = ortho_group.rvs(dim = p)
  V = V[0:p,0:d]
  print("V.shape= ", V.shape)
  
  lamb = np.matmul(V,lamb_lat)  
  
  
  #etas = np.matrix(t.rvs(10, 0, 1, size = npc).reshape(npc,1))
  #print(etas)
  #etas_1 = np.matrix(t.rvs(10, -1.5, 0.5, size = npc).reshape(npc,1))
  #etas_1 = np.random.normal(loc = -1.5, scale = 0.5, size = (npc,1))
  etas_1 = np.empty((npc,2))
  etas_1[:,0] = np.random.normal(loc = -2, scale = 0.1, size = npc)
  etas_1[:,1] = np.random.normal(loc = 0, scale = 0.1, size = npc)
  
  #etas_2 = np.matrix(t.rvs(10, 1.5, 0.5, size = npc).reshape(npc,1))
  #etas_2 = np.random.normal(loc = 1.5, scale = 0.5, size = (npc,1))
  etas_2 = np.empty((npc,2))
  etas_2[:,0] = np.random.normal(loc = 2, scale = 0.1, size = npc)
  etas_2[:,1] = np.random.normal(loc = 0, scale = 0.1, size = npc)
  
  etas = np.concatenate((etas_1,etas_2))
  #print(etas)
  
  data = np.matmul(lamb,etas.T).T +  np.vstack([mvn.rvs(cov = np.diag(sigma_bar_cov)) for i in range(etas.shape[0])])
  
  #np.savetxt('data/Free_svd_lambda_multidim/applam/dataset.csv', data, delimiter=',', fmt='%f')
  
  
  
  # centering of data
  col_mean =np.mean(data,axis = 0)
  print("column means: ", col_mean)
  
  data_zero_mean = data - col_mean
  
  data_scaled= data
  
  U, S, Vh = np.linalg.svd(data_zero_mean, full_matrices=True)
  
  lamb_est = Vh[0:d, ].transpose()
  
  #print("lambda_est.shape: ", lamb_est.shape) 
  
  #print("Lambda:", lamb_est) 
  
  
  
  

  
  
  
  
    
  
  
  
  