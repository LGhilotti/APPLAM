# APPLAM
This repository contains the code for running the block Gibbs-sampler designed for the APPLAM model; additionally, it contains data and code for replicating simulations and application.

### Example for running the algorithm

#############################################
### IMPORT LIBRARIES AND FUNCTIONS ##########
#############################################

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
from itertools import product

import sys
sys.path.append('.')
sys.path.append('./pp_mix')

from pp_mix.interface import ConditionalMCMC, cluster_estimate
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params

##############################################
# COMMON QUANTITIES TO ALL RUNS OF ALGORITHM #
##############################################

# Set hyperparameters (fixing the covariance matrices equal to the identity, both Sigma e Deltas)
params_file = "data/Bauges_data/resources/sampler_params_fixed.asciipb" # substitute with your own file with identical structure

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =1000
nburn=10000
niter = 4000
thin= 5
log_ev=100

# Set dimension of the latent space
d = 3

################################################
############### DATA ###########################
################################################

# Upload your data in n x p matrix in variable "data"

####################################
##### HYPERPARAMETERS ##############
####################################

# Set the expected number of centers a priori
rho = 30

# Fix "s", then: rho_max = rho/s
# It follows: c = rho_max * (2 pi)^{d/2}
s = 0.5
rho_max = rho/s
c = rho_max * ((2. * np.pi) ** (float(d)/2))

hyperpar = Params()
with open(params_file, 'r') as fp:
    text_format.Parse(fp.read(), hyperpar)
    
hyperpar.dpp.c = c
hyperpar.dpp.n = n
hyperpar.dpp.s = s

hyperpar.wishart.nu = hyperpar.wishart.nu + d

# hyperquare side length
sidelength = 30

###################################
######## MCMC SAMPLER #############
###################################

# Build the sampler
sampler = ConditionalMCMC(hyperpar = hyperpar)

# Run the algorithm
sampler.run(ntrick, nburn, niter, thin, data, d, sidelength, log_every = log_ev)
# with no ranges should return ERROR
#sampler.run_binary(ntrick, nburn, niter, thin, data, d, log_every = log_ev)
