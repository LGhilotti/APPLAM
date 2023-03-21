library(AntMAN)
library(BNPmix)
library(coda)


fit_antman_multi <- function(data) {
  mcmc_params = AM_mcmc_parameters(niter=50000, burnin=25000, thin=5,
                                   output = c("CI","K","M","Tau","H","Q","S"))
  
  d = dim(data)[2]
  mixture_mvn_params = AM_mix_hyperparams_multinorm(mu0=rep(0, d), ka0=.5, nu0=4, Lam0=diag(d))
  components_prior = AM_mix_components_prior_pois (init=4, a=10, b=12)
  weights_prior = AM_mix_weights_prior_gamma(init=2, a=1, b=1)
  fit <- AM_mcmc_fit(
    y = as.matrix(data),
    mix_kernel_hyperparams = mixture_mvn_params,
    mix_components_prior = components_prior,
    mix_weight_prior = weights_prior,
    mcmc_parameters = mcmc_params)
  fit
}


fit_antman_univ <- function(data) {
  mixture_uni_params = AM_mix_hyperparams_uninorm(m0=0, k0=.1, nu0=1, sig02=1)
  mcmc_params = AM_mcmc_parameters(niter=50000, burnin=25000, thin=5,
                                   output = c("CI","K","M","Tau","H","Q","S"))
  components_prior = AM_mix_components_prior_pois (init=10, a=2, b=0.5)
  weights_prior = AM_mix_weights_prior_gamma(a=2, b=2)
  fit <- AM_mcmc_fit(
    y = data,
    mix_kernel_hyperparams = mixture_uni_params,
    mix_components_prior = components_prior,
    mix_weight_prior = weights_prior,
    mcmc_parameters = mcmc_params)
  fit
}



fit_bnpmix <- function(data, grid) {
  est_model <- PYdensity(y = data, mcmc = list(niter = 20000, nburn = 10000, method="MAR", hyper=F),
                         prior = list(k0 = 0.1, strength = 1, discount = 0.0),
                         output = list(grid = grid))
  est_model
}

fit_bnpmix_noest <- function(data) {
  est_model <- PYdensity(y = data, mcmc = list(niter = 20000, nburn = 10000, method="MAR", hyper=F),
                         prior = list(k0 = 0.1, strength = 1, discount = 0.0))
  est_model
}


##################################
### MIXTURE OF T AND SKEW-NORMAL #
##################################

data_files = c(
  "data/sim_t_skew_dim1.txt",
  "data/sim_t_skew_dim2.txt",
  "data/sim_t_skew_dim5.txt",
  "data/sim_t_skew_dim10.txt")

# Univariate Case
data = read.csv(data_files[1], sep=" ", header = F)$V1
fit <- fit_antman_univ(data)
write.table(fit$K, "data/sim_t_skew_dim1_Kraf.txt", sep=" ", row.names=F, col.names=F)

grid <- seq(-10, 10, length.out = 100)
est_model <- fit_bnpmix(data, grid)
nclus = apply(est_model$clust, 1, max)
write.table(nclus, "data/sim_t_skew_dim1_Kbnpmix.txt", sep=" ", row.names=F, col.names=F)

# Multivariate Cases
data = as.matrix(read.csv(data_files[2], sep=" ", header = F))
fit <- fit_antman_multi(data)
write.table(fit$K, "data/sim_t_skew_dim2_Kraf.txt", sep=" ", row.names=F, col.names=F)

est_model <- fit_bnpmix_noest(data)
nclus = apply(est_model$clust, 1, max)
write.table(nclus, "data/sim_t_skew_dim2_Kbnpmix.txt", sep=" ", row.names=F, col.names=F)


#########################
####  MULTIVARIATE 1 ####
#########################

data = read.csv("data/data_0.txt", sep=" ", header = F)

fit <- fit_antman_multi(data)
saveRDS(fit, "data/data_0_fit_antman.RData")
write.table(fit$M, "data/data_0_mtot_raf.txt", sep=" ", row.names=F, col.names=F)
write.table(fit$K, "data/data_0_K_raf.txt", sep=" ", row.names=F, col.names=F)

grid <- expand.grid(seq(-10, 10, length.out = 100), seq(-10, 10, length.out = 100))
est_model <- fit_bnpmix(data, grid)
# saveRDS(est_model, "data/data_0_fit_bnpmix.RData")
plot(est_model)

write.table(colMeans(est_model$density), "data/data_1_dens_bnpmix.txt", sep=" ", row.names=F, col.names=F)
nclus = apply(est_model$clust, 1, max)
write.table(nclus, "data/data_0_K_bnpmix.txt", sep=" ", row.names=F, col.names=F)

#########################
####  MULTIVARIATE 2 ####
#########################

data = read.csv("data/data_1.txt", sep=" ", header = F)

fit <- fit_antman(data)
saveRDS(fit, "data/data_1_fit_antman.RData")
summary(fit)
plot(fit)
write.table(fit$M, "data/data_1_mtot_raf.txt", sep=" ", row.names=F, col.names=F)
write.table(fit$K, "data/data_1_K_raf.txt", sep=" ", row.names=F, col.names=F)



grid <- expand.grid(seq(-6, 7, length.out = 100), seq(-6, 7, length.out = 100))
est_model <- fit_bnpmix(data, grid)
saveRDS(est_model, "data/data_1_fit_bnpmix.RData")

write.table(est_model$density, "data/data_1_dens_bnpmix.txt", sep=" ", row.names=F, col.names=F)
nclus = apply(est_model$clust, 1, max)
write.table(nclus, "data/data_1_K_bnpmix.txt", sep=" ", row.names=F, col.names=F)
mean(nclus)

est_model$density

#######################
####  UNIVARIATE 0 ####
#######################

data = read.csv("data/data_univ_0.txt", sep=" ", header = F)
fit <- fit_antman_univ(data$V1)
plot(fit)

saveRDS(fit, "data/data_univ_0_fit_antman.RData")
write.table(fit$M, "data/data_univ_0_mtot_raf.txt", sep=" ", row.names=F, col.names=F)
write.table(fit$K, "data/data_univ_0_K_raf.txt", sep=" ", row.names=F, col.names=F)



grid <- seq(-10, 10, length.out=200)
est_model <- fit_bnpmix(data$V1, grid)
plot(est_model)
saveRDS(est_model, "data/data_0_univ_fit_bnpmix.RData")

write.table(est_model$density, "data/data_0_univ_dens_bnpmix.txt", sep=" ", row.names=F, col.names=F)
nclus = apply(est_model$clust, 1, max)
write.table(nclus, "data/data_0_univ_K_bnpmix.txt", sep=" ", row.names=F, col.names=F)
mean(nclus)

#######################
####  UNIVARIATE 1 ####
#######################

data = read.csv("data/data_univ_1.txt", sep=" ", header = F)
fit <- fit_antman_univ(data$V1)
plot(fit)

saveRDS(fit, "data/data_univ_1_fit_antman.RData")
write.table(fit$M, "data/data_univ_1_mtot_raf.txt", sep=" ", row.names=F, col.names=F)
write.table(fit$K, "data/data_univ_1_K_raf.txt", sep=" ", row.names=F, col.names=F)



grid <- seq(-10, 10, length.out=200)
est_model <- fit_bnpmix(data$V1, grid)
plot(est_model)
saveRDS(est_model, "data/data_1_univ_fit_bnpmix.RData")

write.table(est_model$density, "data/data_1_univ_dens_bnpmix.txt", sep=" ", row.names=F, col.names=F)
nclus = apply(est_model$clust, 1, max)
write.table(nclus, "data/data_1_univ_K_bnpmix.txt", sep=" ", row.names=F, col.names=F)
mean(nclus)


###################
####  BANANA 1 ####
###################

data = read.csv("data/banana_data.txt", sep=" ", header = F)

components_prior = AM_mix_components_prior_pois (init=5, a=10, b=2)
weights_prior = AM_mix_weights_prior_gamma(init=2, a=1, b=1)
AM_multinorm_mix_hyperparams(mu0=c(0,0), ka0=.5, nu0=4, Lam0=diag(2))

mcmc_params = AM_mcmc_parameters(niter=50000, burnin=25000, thin=5,
                                 output = c("CI","K","M","Tau","H","Q","S"))
fit <- AM_mcmc_fit(
  y = as.matrix(data),
  mix_kernel_hyperparams = mixture_mvn_params,
  mix_components_prior = components_prior,
  mix_weight_prior = weights_prior,
  mcmc_parameters = mcmc_params)

cols = fit$CI[[length(fit$CI)]]
plot(data[, 1], data[, 2], col=cols + 1)
hist(fit$K)


grid <- expand.grid(seq(-2, 2, length.out = 100), seq(-2, 2, length.out = 100))
est_model <- PYdensity(y = data, mcmc = list(niter = 20000, nburn = 10000, method="MAR", hyper=F),
                       prior = list(k0 = 0.5, strength = 1, discount = 0.0),
                       output = list(grid = grid))
plot(est_model)
nclus = apply(est_model$clust, 1, max)
hist(nclus)


######################
### MILLER DUNSON ####
######################


data = read.csv("data/data_univ_miller_dunson.txt", sep=" ", header = F)$V1
hist(data)
fit <- fit_antman_univ(data)
hist(fit$K)
# plot(fit)

saveRDS(fit, "data/data_univ_miller_dunson.fit_antman.RData")
write.table(fit$M, "data/ddata_univ_miller_dunson_mtot_raf.txt", sep=" ", row.names=F, col.names=F)
write.table(fit$K, "data/data_univ_miller_dunson_K_raf.txt", sep=" ", row.names=F, col.names=F)

grid <- seq(-10, 10, length.out=200)
est_model <- fit_bnpmix(data, grid)
plot(est_model)
saveRDS(est_model, "data/data_univ_miller_dunson._fit_bnpmix.RData")

write.table(est_model$density, "data/data_univ_miller_dunson_dens_bnpmix.txt", sep=" ", row.names=F, col.names=F)
nclus = apply(est_model$clust, 1, max)
hist(nclus)
write.table(nclus, "data/data_univ_miller_dunson_K_bnpmix.txt", sep=" ", row.names=F, col.names=F)
mean(nclus)
