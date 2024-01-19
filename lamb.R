library(Rcpp)

sourceCpp("DL_linear_split_merge_package.cpp")

DL_mixture_function <- function(y, nrun_lamb = 10, burn_lamb = 2,  thin_lamb = 1,
      a_s = 1, b_s = 0.3, a = 0.5, diag_psi_iw=20, niw_kap=1e-3, a_dir=0.1, b_dir=0.1, seed=1234){

  set.seed(seed)  
  
  n = nrow(y)
  
  # 4.2) Empirical Estimation the Latent Dimension d and Initializing eta & lambda
  pca.results=irlba::irlba(y, nv=40)
  
  # d is set to be the minimum number of eigenvalues explaining at least 95% of the variability in the data.
  cum_eigs= cumsum(pca.results$d)/sum(pca.results$d)
  d_lamb=min(which(cum_eigs>.95))
  print(paste0("latent dimension: ",d_lamb))
  
  # Left and right singular values of y are used to initialize eta and lambda respectively for the MCMC.
  eta= pca.results$u %*% diag(pca.results$d)
  eta=eta[,1:d_lamb] #Left singular values of y are used as to initialize eta
  lambda=pca.results$v[,1:d_lamb] #Right singular values of y are used to initialize lambda
  
  # 4.3) Initializing Cluster Allocations [IMPACTFULL]
  cluster.start = rep(0,n)
  #cluster.start = kmeans(y, 2)$cluster
  
  # 4.4) Set Prior Parameter of Lamb
  nu=d_lamb+50
  
  result.lamb <- DL_mixture(a_dir=a_dir, b_dir=b_dir, diag_psi_iw=diag_psi_iw, niw_kap=niw_kap, niw_nu=nu,
                          as=a_s, bs=b_s, a=a,
                          nrun=nrun_lamb, burn=burn_lamb, thin=thin_lamb,
                          nstep = 5, prob = 0.5,
                          lambda, eta, y,
                          del = cluster.start,
                          dofactor=1 )
                          
  burn= burn_lamb/thin_lamb
  post.samples=result.lamb[-(1:burn),]+1
                          
  return (post.samples)
  

}