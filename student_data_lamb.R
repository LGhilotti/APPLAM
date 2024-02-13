library("Rcpp")
library("readr")
library("irlba")
library("readxl")
library("iterators")
library("glue")

sourceCpp("DL_linear_split_merge_package.cpp") ##This is the souce C++ file


#################
set.seed(1234)

nburn=0
niter =1e5
thin= 10


############################################
## RUN ON DIFFERENT DATASETS AND SETTINGS ##
############################################

p_s = c(400)
#p_s = 400
d_s = c(5)
#d_s = 10
#M_s = c(4, 8, 12)
M_s = 4
n_percluster_s = c(50)

# Outer cycle for reading the different datasets and perform the estimation
for (p in p_s){
  for (dtrue in d_s){
    for (M in M_s){
      for (npc in n_percluster_s){
        #######################################
        ### READ DATA AND PRE-PROCESSING ######
        #######################################

        # read the dataset
        data = read_csv(file = glue("data/Student_latent_data/datasets/stud_p_{p}_d_{dtrue}_M_{M}_npc_{npc}_data.csv") , col_names = FALSE)

        # scaling of data
        centering.var=median(colMeans(data))
        scaling.var=median(apply(data,2,sd))
        data_scaled=as.matrix((data-centering.var)/scaling.var)

        # Read the latent dimension for the current datasets (already computed since used also in Applam)
        #d = strtoi(read_file(file = glue("data/Student_data/latent_dim/stud_p_{p}_d_{dtrue}_M_{M}_npc_{npc}_lat_dim.txt")))
        d = dtrue

        #outpath_d = glue("data/Student_latent_data/lamb_out/lamb_p_{p}_d_{dtrue}_M_{M}_npc_{npc}_out")
        #if (!(dir.exists(outpath_d))){
        #  dir.create(outpath_d)
        #}

        # Initialization eta and lambda
        pca.results=irlba::irlba(data_scaled, nv=10)
        eta= pca.results$u %*% diag(pca.results$d)
        eta=eta[,1:d] #Left singular values of y are used as to initialize eta
        lambda=pca.results$v[,1:d] #Right singular values of y are used to initialize lambda

        # Initialization cluster allocations
        cluster.start = kmeans(data_scaled, 80)$cluster - 1

        ### Set Prior Parameter of `Lamb`
        as = 1; bs = 0.3
        a=0.5
        diag_psi_iw=20
        niw_kap=1e-3
        nu=d+50

        #conc_dir_s = c(0.1,0.5,1)
        conc_dir_s = c(0.5)

        for (conc_dir in conc_dir_s){
          #### Fit the `Lamb` Model
          result.lamb <- DL_mixture(conc_dir, diag_psi_iw=diag_psi_iw, niw_kap=niw_kap, niw_nu=nu,
                                    as=as, bs=bs, a=a,
                                    nrun=niter, burn=nburn, thin=thin,
                                    nstep = 5,prob=0.5, #With probability `prob` either the Split-Merge sampler with `nstep` Gibbs scans or Gibbs sampler in performed
                                    lambda, eta, data_scaled,
                                    del = cluster.start,
                                    dofactor=1 #If dofactor set to 0, clustering will be done on the initial input eta values only; eta will not be updated in MCMC
          )

          # Save results in folder
          #base_outpath_conc = outpath_d + "/conc_{conc_dir}_out"

          #i=1
          #while (dir.exists(base_outpath_conc + "{i}")){
          #  i = i+1
          #}
          #outpath = base_outpath_conc + "{i}"
          #dir.create(outpath)

          disc=nburn/thin
          post.samples=result.lamb[-(1:disc),]+1
          #write.table(post.samples, file = outpath + "/alloc_matrix.csv" ,
          #            quote=FALSE, eol="\n", row.names=FALSE, col.names=FALSE,  sep=",")
          write.table(post.samples, file = "data/alloc_matrix_noburn.csv" ,
                      quote=FALSE, eol="\n", row.names=FALSE, col.names=FALSE,  sep=",")
          #write(x=conc_dir, file = outpath + "/conc_param.txt")


        }


      }
    }
  }
}





