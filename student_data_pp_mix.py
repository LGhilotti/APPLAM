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
from itertools import product

import sys
sys.path.append('.')
sys.path.append('./pp_mix')

from pp_mix.interface import ConditionalMCMC, ConditionalMCMC_isotropic, cluster_estimate
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params

np.random.seed(12345)

##############################################
# COMMON QUANTITIES TO ALL RUNS OF ALGORITHM #
##############################################

# Set hyperparameters (agreeing with Chandra)
DEFAULT_PARAMS_FILE = "data/Student_data/resources/sampler_params.asciipb"
SPECIFIC_PARAMS_FILE = "data/Student_data/resources/pars_p_{0}_d_{1}_M_{2}_npc_{3}.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =1000
nburn=2000
niter = 4000
thin= 5
log_ev=100

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_values", nargs="+", default=["100", "200", "400"])
    parser.add_argument("--d_values", nargs="+", default=["2","5","8"])
    parser.add_argument("--m_values", nargs="+", default=["4"])
    parser.add_argument("--n_by_clus", nargs="+", default=["50"])
    args = parser.parse_args()

    p_s = list(map(int, args.p_values))
    d_s = list(map(int, args.d_values))
    M_s = list(map(int, args.m_values))
    n_percluster_s = list(map(int, args.n_by_clus))


    # Outer cycle for reading the different datasets and perform the estimation
    for p,dtrue,M,npc in product(p_s, d_s, M_s, n_percluster_s):
        
        #######################################
        ### READ DATA AND PRE-PROCESSING ######
        #######################################
        
        # read the dataset
        with open("data/Student_data/datasets/stud_p_{0}_d_{1}_M_{2}_npc_{3}_data.csv".format(p,dtrue,M,npc), newline='') as my_csv:
            data = pd.read_csv(my_csv, sep=',', header=None).values
            
        # scaling of data
        centering_var=stat.median(np.mean(data,0))
        scaling_var=stat.median(np.std(data,0))
        data_scaled=(data-centering_var)/scaling_var

        # Read the latent dimension for the current datasets (already computed since used also in Chandra)
        #with open("data/Student_data/latent_dim/stud_p_{0}_d_{1}_M_{2}_npc_{3}_lat_dim.txt".format(p,dtrue,M,npc)) as my_file:
        #    d = int(my_file.read())
        d = dtrue
        
        outpath_d = "data/Student_data/applam/app_p_{0}_d_{1}_M_{2}_npc_{3}_out".format(p,dtrue,M,npc)
        if not(os.path.exists(outpath_d)):
            os.makedirs(outpath_d)

        ####################################
        ##### HYPERPARAMETERS ##############
        ####################################

        # Set the expected number of centers a priori
        rho_s = [5,10,20]
        
        for rho in rho_s:
            
            # Fix "s", then: rho_max = rho/s
            # It follows: c = rho_max * (2 pi)^{d/2}
            s = 0.5
            rho_max = rho/s
            c = rho_max * ((2. * np.pi) ** (float(d)/2))
    
            hyperpar = Params()
            params_file = SPECIFIC_PARAMS_FILE.format(p,dtrue,M,npc)
            if os.path.exists(params_file):
                print("Using dataset-specific params file for "
                      "'p'={0}, 'd'={1} 'M'={2}, 'npc'={3}".format(p,dtrue,M,npc))
            else:
                print("Using default params file for "
                      "'p'={0}, 'd'={1} 'M'={2}, 'npc'={3}".format(p,dtrue,M,npc))
                params_file = DEFAULT_PARAMS_FILE
            with open(params_file, 'r') as fp:
                text_format.Parse(fp.read(), hyperpar)

            hyperpar.dpp.c = c
            hyperpar.dpp.n = n
            hyperpar.dpp.s = s
            
            hyperpar.wishart.nu = hyperpar.wishart.nu + d

            #################################################
            ######## MCMC SAMPLER - ANISOTROPIC #############
            #################################################

            # Build the sampler
            sampler_aniso = ConditionalMCMC(hyperpar = hyperpar)

            # Run the algorithm
            sampler_aniso.run(ntrick, nburn, niter, thin, data_scaled, d, log_every = log_ev)


            # Save results in folder
            base_outpath_rho = os.path.join(outpath_d, "rho_{0}_out".format(rho)) + "_{0}"
            i = 0
            while os.path.exists(base_outpath_rho.format(i)):
                i = i+1
            outpath = base_outpath_rho.format(i)
            os.makedirs(outpath)
                
            # Save the serialized chain produced by the sampler
            sampler_aniso.serialize_chains(os.path.join(outpath, "chains.recordio"))


            # save the parameters
            with open(os.path.join(outpath, "params.asciipb"), 'w') as fp:
                fp.write(text_format.MessageToString(hyperpar))


            chain_aniso = sampler_aniso.chains

            # plots
            fig = plt.figure()
            tau_chain_aniso = np.array([x.lamb_block.tau for x in chain_aniso])
            plt.plot(tau_chain_aniso)
            plt.title("tau chain - APPLAM")
            plt.savefig(os.path.join(outpath, "tau_chain_aniso.pdf"))
            plt.close()

            fig = plt.figure()
            first_sbar_chain_aniso = np.array([to_numpy(x.sigma_bar)[0] for x in chain_aniso])
            plt.plot(first_sbar_chain_aniso,color='red')
            last_sbar_chain_aniso = np.array([to_numpy(x.sigma_bar)[-1] for x in chain_aniso])
            plt.plot(last_sbar_chain_aniso,color='blue')
            plt.title("sbar_chain - APPLAM")
            plt.savefig(os.path.join(outpath, "sbar_chain_aniso.pdf"))
            plt.close()

            # Compute Posterior Summaries
            fig = plt.figure()
            n_cluster_chain_aniso = np.array([x.ma for x in chain_aniso])
            plt.plot(n_cluster_chain_aniso)
            plt.title("number of clusters chain - APPLAM")
            plt.savefig(os.path.join(outpath, "nclus_chain_aniso.pdf"))
            plt.close()

            #fig = plt.figure()
            n_nonall_chain_aniso = np.array([x.mna for x in chain_aniso])
            #plt.plot(n_nonall_chain)
            #plt.title("number of non allocated components chain")
            #plt.savefig(os.path.join(outpath, "non_alloc_chain.pdf"))
            #plt.close()

            post_mode_nclus_aniso = mode(n_cluster_chain_aniso)[0][0] # store in dataframe
            post_avg_nclus_aniso = n_cluster_chain_aniso.mean() # store in dataframe
            post_avg_nonall_aniso =  n_nonall_chain_aniso.mean() # store in dataframe

            clus_alloc_chain_aniso = [x.clus_alloc for x in chain_aniso]
            best_clus_aniso = cluster_estimate(np.array(clus_alloc_chain_aniso))
            np.savetxt(os.path.join(outpath, "best_clus_aniso.txt"), best_clus_aniso)

            n_clus_best_clus_aniso = np.size(np.unique(best_clus_aniso))
            true_clus = np.repeat(range(M),npc)
            ari_best_clus_aniso = adjusted_rand_score(true_clus, best_clus_aniso) # store in dataframe
            aris_chain_aniso = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain_aniso])
            mean_aris_aniso, sigma_aris_aniso = np.mean(aris_chain_aniso), np.std(aris_chain_aniso) # store mean_aris in dataframe
            CI_aris_aniso = norm.interval(0.95, loc=mean_aris_aniso, scale=sigma_aris_aniso/sqrt(len(aris_chain_aniso))) # store in dataframe
            list_performance = list()
            list_performance.append([p,dtrue,d,M,npc,sampler.means_ar, sampler.lambda_ar, rho, post_mode_nclus,
                                post_avg_nclus, post_avg_nonall, ari_best_clus, CI_aris])
            df_performance = pd.DataFrame(list_performance, columns=('p','dtrue','d','M','npc','means_ar','lambda_ar', 'intensity',
                                                'mode_nclus', 'avg_nclus', 'avg_nonalloc', 'ari_best_clus', 'CI_aris'))
            df_performance.to_csv(os.path.join(outpath, "df_performance.csv"))

