# K-Nearest Neighbors Time Series Prediction with Invariances

KNN-TSPI python and R implementation, the full description of the algorithm is available at: https://www.researchgate.net/publication/300414605_A_Study_of_the_Use_of_Complexity_Measures_in_the_Similarity_Search_Process_Adopted_by_kNN_Algorithm_for_Time_Series_Prediction

## Parameter description
### Python
- k : number of neighbors;
- len_query : number of observations in each query;
- weights : if "uniform" the prediction is given by the mean, if "distance" then a custom function g that takes the distances and outputs its corresponding weights is applied. The default is the inverse of the distance;
- h : forecasting horizon.
### R
- k : number of neighbors;
- len_query : number of observations in each query;
- target : way to combine the labels from the k nearest neighbors, it can be either "mean", "median" or "custom". The latter applies a given function g that takes the distances as input and outputs its corresponding weights, the default is the inverse of the distance;
- pred_interval : whether to calculate the 80% and 95% prediction intervals using bootstrapping, it can be quite time consuming;
- h : forecasting horizon.
 
## Examples
### Python

```python
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.datasets import load_woolyrnq
from pmdarima.model_selection import RollingForecastCV
from knn_tspi import KNeighborsTSPI


# Loads the time series
data = load_woolyrnq()

# Sets the knn-tspi hyperparameters
k = 3
len_query = 4
h = 16

# Calculates the model residuals using TSCV
res = []
for train_idxs, test_idx in RollingForecastCV(h=1, step=1, initial=3*len_query).split(data):
    knn = KNeighborsTSPI(k=k, len_query=len_query)
    knn.fit(data[train_idxs])
    res.append(data[test_idx][0]-knn.predict(h=1)[0])

# Forecast the future observations
knn = KNeighborsTSPI(k=k, len_query=len_query)
knn.fit(data)
y = knn.predict(h=h)

# Applies bootstrapping to estimate the forecasting intervals
futures = []
for i in range(1000):
    futures.append(y+np.random.choice(res, h))
futures = np.array(futures)
intervals = np.quantile(futures, q=[0.2, 0.8, 0.15, 0.95], axis=0)

# Plots the results
rng = range(len(data), len(data)+h)
plt.title("KNN-TSPI Predictions - Quarterly production of woollen yarn in Australia")
plt.plot(data, color="black", label="Historical")
plt.plot(rng, y, color="blue", label="Mean")
plt.fill_between(
    rng, intervals[0, :], intervals[1, :], color="cornflowerblue", label="80% confidence")
plt.fill_between(
    rng, intervals[1, :], intervals[3, :], color="lightsteelblue", label="95% confidence")
plt.fill_between(rng, intervals[2, :], intervals[0, :], color="lightsteelblue")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Y")
plt.legend(loc="lower left")
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/56834802/108731205-186fc400-750b-11eb-97ab-32e739096c5a.png)

### R

```R
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
```
| Low 95 | Low 80 | High 80 | High 95 | Mean |
| ------ | ------ | ------- | ------- | ---- |
| 4431.294 | 5141.181 | 6001.490 | 6431.195 | 5600.233 |
| 4247.463 | 4995.754 | 5732.670 | 6091.139 | 5416.403 |
| 4707.657 | 5382.555 | 6245.636 | 6707.558 | 5876.596 |
| 4636.162 | 5383.324 | 6211.415 | 6590.123 | 5842.375 |
| 4451.910 | 5192.074 | 5995.703 | 6489.085 | 5658.124 |
| 4228.253 | 5057.400 | 5808.576 | 6214.364 | 5470.996 |
| 4548.144 | 5260.317 | 6091.937 | 6497.726 | 5754.358 |
| 4656.614 | 5422.569 | 6250.660 | 6712.582 | 5881.621 |
| 4533.354 | 5299.308 | 6127.400 | 6501.728 | 5758.360 |
| 4240.715 | 5006.669 | 5834.761 | 6140.457 | 5465.721 |
| 4439.815 | 5104.407 | 5946.334 | 6352.122 | 5608.754 |
| 4599.052 | 5330.018 | 6161.638 | 6655.020 | 5824.058 |
| 4782.719 | 5323.383 | 6108.720 | 6478.585 | 5735.217 |
| 4237.821 | 4968.787 | 5831.867 | 6206.196 | 5462.828 |
| 4433.795 | 5129.041 | 5977.589 | 6383.377 | 5640.009 |
| 4566.425 | 5297.390 | 6129.010 | 6466.167 | 5791.431 |
