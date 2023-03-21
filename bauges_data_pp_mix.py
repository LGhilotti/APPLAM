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

from pp_mix.interface import ConditionalMCMC, cluster_estimate
from pp_mix.utils import loadChains, to_numpy, to_proto
from pp_mix.protos.py.state_pb2 import MultivariateMixtureState, EigenVector, EigenMatrix
from pp_mix.protos.py.params_pb2 import Params

np.random.seed(12345)

##############################################
# COMMON QUANTITIES TO ALL RUNS OF ALGORITHM #
##############################################

# Set hyperparameters (fixing the covariance matrices equal to the identity, both Sigma e Deltas)
DEFAULT_PARAMS_FILE = "data/Bauges_data/resources/sampler_params_fixed.asciipb"
SPECIFIC_PARAMS_FILE = "data/Bauges_data/resources/pars_d_{0}_fixed.asciipb"

# Set the truncation level N (here called n)
n = 3

# Set sampler parameters
ntrick =1000
nburn=10000
niter = 4000
thin= 5
log_ev=100

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_values", nargs="+", default=["3"])
    args = parser.parse_args()

    d_s = list(map(int, args.d_values))


    #######################################
    ### READ DATA AND PRE-PROCESSING ######
    #######################################

    # read the dataset
    with open("data/Bauges_data/Bauges_data.csv", newline='') as my_csv:
            data_names = pd.read_csv(my_csv, sep=',', index_col=0)

    data = data_names.values

    for d in d_s:

        outpath_d = "data/Bauges_data/applam_d_{0}_fixed_moreclusters_out".format(d)
        if not(os.path.exists(outpath_d)):
            os.makedirs(outpath_d)

        ####################################
        ##### HYPERPARAMETERS ##############
        ####################################

        # Set the expected number of centers a priori
        rho_s = [30]

        for rho in rho_s:

            # Fix "s", then: rho_max = rho/s
            # It follows: c = rho_max * (2 pi)^{d/2}
            s = 0.5
            rho_max = rho/s
            c = rho_max * ((2. * np.pi) ** (float(d)/2))

            hyperpar = Params()
            params_file = SPECIFIC_PARAMS_FILE.format(d)
            if os.path.exists(params_file):
                print("Using dataset-specific params file for "
                      "'d'={0}".format(d))
            else:
                print("Using default params file for "
                      "'d'={0}".format(d))
                params_file = DEFAULT_PARAMS_FILE
            with open(params_file, 'r') as fp:
                text_format.Parse(fp.read(), hyperpar)

            hyperpar.dpp.c = c
            hyperpar.dpp.n = n
            hyperpar.dpp.s = s

            #hyperpar.wishart.nu = hyperpar.wishart.nu + d

            # different hyperquares side length
            sidelength_s = [30]

            for sidelength in sidelength_s:

                ###################################
                ######## MCMC SAMPLER #############
                ###################################

                # Build the sampler
                sampler = ConditionalMCMC(hyperpar = hyperpar)

                # Run the algorithm
                sampler.run_binary(ntrick, nburn, niter, thin, data, d, sidelength, log_every = log_ev)
                # with no ranges should return ERROR
                #sampler.run_binary(ntrick, nburn, niter, thin, data, d, log_every = log_ev)

                # Save results in folder
                base_outpath_rho = os.path.join(outpath_d, "rho_{0}_ranges_{1}_out".format(rho,sidelength)) + "_{0}"
                i = 0
                while os.path.exists(base_outpath_rho.format(i)):
                    i = i+1
                outpath = base_outpath_rho.format(i)
                os.makedirs(outpath)

                # Save the serialized chain produced by the sampler
                sampler.serialize_chains(os.path.join(outpath, "chains.recordio"))


                # save the parameters
                with open(os.path.join(outpath, "params.asciipb"), 'w') as fp:
                    fp.write(text_format.MessageToString(hyperpar))


                chain = sampler.chains

                # plots
                fig = plt.figure()
                tau_chain = np.array([x.lamb_block.tau for x in chain])
                plt.plot(tau_chain)
                plt.title("tau chain")
                plt.savefig(os.path.join(outpath, "tau_chain.pdf"))
                plt.close()

                # IN FIXED VERSION SIGMA_BAR IS NOT UPDATED
                #fig = plt.figure()
                #first_sbar_chain = np.array([to_numpy(x.sigma_bar)[0] for x in chain])
                #plt.plot(first_sbar_chain,color='red')
                #last_sbar_chain = np.array([to_numpy(x.sigma_bar)[-1] for x in chain])
                #plt.plot(last_sbar_chain,color='blue')
                #plt.title("sbar_chain")
                #plt.savefig(os.path.join(outpath, "sbar_chain.pdf"))
                #plt.close()


                # Compute Posterior Summaries
                fig = plt.figure()
                n_cluster_chain = np.array([x.ma for x in chain])
                plt.plot(n_cluster_chain)
                plt.title("number of clusters chain")
                plt.savefig(os.path.join(outpath, "nclus_chain.pdf"))
                plt.close()

                fig = plt.figure()
                n_nonall_chain = np.array([x.mna for x in chain])
                plt.plot(n_nonall_chain)
                plt.title("number of non allocated components chain")
                plt.savefig(os.path.join(outpath, "non_alloc_chain.pdf"))
                plt.close()

                post_mode_nclus = mode(n_cluster_chain)[0][0] # store in dataframe
                post_avg_nclus = n_cluster_chain.mean() # store in dataframe
                post_avg_nonall =  n_nonall_chain.mean() # store in dataframe

                clus_alloc_chain = [x.clus_alloc for x in chain]
                best_clus = cluster_estimate(np.array(clus_alloc_chain))
                np.savetxt(os.path.join(outpath, "best_clus.txt"), best_clus)

                unique_bc, counts_bc = np.unique(best_clus, return_counts=True)

                num_best_clus = np.size(unique_bc)
                num_sigleton_cl = np.sum(counts_bc==1)

                #true_clus = np.repeat(range(M),npc)
                #ari_best_clus = adjusted_rand_score(true_clus, best_clus) # store in dataframe
                #aris_chain = np.array([adjusted_rand_score(true_clus, x) for x in clus_alloc_chain])
                #mean_aris, sigma_aris = np.mean(aris_chain), np.std(aris_chain) # store mean_aris in dataframe
                #CI_aris = norm.interval(0.95, loc=mean_aris, scale=sigma_aris/sqrt(len(aris_chain))) # store in dataframe
                list_performance = list()
                list_performance.append([d,sampler.means_ar, sampler.lambda_ar, rho, post_mode_nclus,
                                    post_avg_nclus, post_avg_nonall, num_best_clus, num_sigleton_cl])
                df_performance = pd.DataFrame(list_performance, columns=('d','means_ar','lambda_ar', 'intensity',
                                                    'mode_nclus', 'avg_nclus', 'avg_nonalloc', 'num_best_clus','num_sigleton_cl'))
                df_performance.to_csv(os.path.join(outpath, "df_performance.csv"))
