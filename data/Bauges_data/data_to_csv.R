load("Bauges_data.Rdata")

write.csv(Bauges_data[,7:(ncol(Bauges_data)-2)],"Bauges_data.csv", 
          row.names = TRUE, col.names = TRUE)


indexes = c(2:6, ncol(Bauges_data)-1, ncol(Bauges_data))

write.csv(Bauges_data[,indexes],"Bauges_data_covariates.csv", 
          row.names = TRUE, col.names = TRUE)
