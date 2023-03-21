# APPLAM
This repository contains the code for running the block Gibbs-sampler designed for the APPLAM model; additionally, it contains data and code for replicating simulations and application.

## Prerequisites

The code was developed and tested on an Ubuntu 18.04 machine and runs in Python3, make sure Python3 is installed. The following instructions lead to successful installation on Linux machines and should be identical for macOS. 
For Windows users: things should work as well modulo the fact that you should be able to install Python packages, have a valid and reasonably new C++ toolchain installed (gcc >= 7.5) and can successfully install and compile the following dependencies.

1. Protobuf: to install the protocol buffer library (C++ runtime library and protoc compiler) follow the [instructions](https://github.com/protocolbuffers/protobuf/tree/master/src). Make sure that the installed library is visible in the PATH (for instance, on Unix machines use `./configure --prefix=/usr`)

2. stan/math: we make extensive use of the awesome math library developed by the Stan team. Simply clone their repo (https://github.com/stan-dev/math) in a local directory and install it. An example of a real instantiation whenever the path to Stan Math is ~/stan-dev/math/:
```shell
make -j4 -f ~/stan-dev/math/make/standalone math-libs
make -f ~/stan-dev/math/make/standalone foo
```
Then set the environmental variable 'STAN_ROOT_DIR' to the path to 'math'.

3. pybind11
```shell
  pip3 install pybind11
```

4. 2to3
```shell
  sudo apt-get install 2to3
  sudo apt-get install python3-lib2to3
  sudo apt-get install python3-toolz
```
## Installation
Installation is trivial on Linux systems and has been tested only on those.
```shell
  cd pp_mix
  make compile_protos
  make generate_pybind
```
and the package is ready to be used!

## Example for running the algorithm

```python
#############################################
### IMPORT LIBRARIES AND FUNCTIONS ##########
#############################################

import numpy as np
import os
import pandas as pd
import statistics as stat

from google.protobuf import text_format

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

#######################################
### READ DATA AND PRE-PROCESSING ######
#######################################

# Upload your data in n x p matrix in variable "data"
# scaling of data
centering_var=stat.median(np.mean(data,0))
scaling_var=stat.median(np.std(data,0))
data_scaled=(data-centering_var)/scaling_var

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

###################################
######## MCMC SAMPLER #############
###################################

# Build the sampler
sampler = ConditionalMCMC(hyperpar = hyperpar)

# Run the algorithm
sampler.run(ntrick, nburn, niter, thin, data_scaled, d, log_every = log_ev)

# Use sampler.serialize_chains(...) to save the serialized chain produced by the sampler
```
