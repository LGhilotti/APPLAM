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
from itertools import product

import sys
sys.path.append('.')
sys.path.append('./pp_mix')

from pp_mix.interface import ConditionalMCMC, cluster_estimate
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params

def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out

# Read chain in recordio format
chain_serialized_file = "data/Bauges_data/applam_d_3_out/rho_0.5_aj_0.001_out/chains_d_3_rho_0.5_aj_0.001.recordio"
objType = MultivariateMixtureState

chain = loadChains(chain_serialized_file, objType)

Lambda_chain = [to_numpy(x.lamb_block.lamb) for x in chain]

Lambda_LambdaT = [np.dot(x, x.T) for x in Lambda_chain]

avg_Lambda_LambdaT = sum(Lambda_LambdaT)/len(Lambda_LambdaT)

np.savetxt("data/Bauges_data/applam_d_3_out/rho_0.5_aj_0.001_out/avg_Lambda_LambdaT.txt", avg_Lambda_LambdaT)

