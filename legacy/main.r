source("knn_tspi.r")

library(forecast)
library(ggplot2)

# Loads the time series
data <- woolyrnq

# Sets the knn-tspi hyperparameters
k <- 3
len_query <- 4
h <- 16

# Makes predictions with forecasting intervals
res <- knn.tspi(
    data=data, k=k, len_query=len_query, h=h, pred_interval=T
)

# Displays results
print(res)
