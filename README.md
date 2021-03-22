# K-Nearest Neighbors Time Series Prediction with Invariances

Python and R implementation of the KNN-TSPI, the full description of the algorithm is available at: https://www.researchgate.net/publication/300414605_A_Study_of_the_Use_of_Complexity_Measures_in_the_Similarity_Search_Process_Adopted_by_kNN_Algorithm_for_Time_Series_Prediction

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
 
## Python Example

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

## R Example

```R
source("knn_tspi.r")

# Reading the dataset
rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat", skip = 1)
rain <- ts(rain, start = 1813, end = 1912, frequency = 1)

# Train test split
h <- 10
train <- tail(rain, -h)
test <- tail(rain, h)

# Model fitting and forecasting
y <- knn.tspi(
    train, 
    k = 5, 
    len_query = 5, 
    target = "median", 
    h = h, 
    pred_interval = T)
print(y)
```
| | Low 95 | Low 80 | High 80 | High 95 | Mean |
|-|------|----|-----|-----|-----|
|1| 15.82227 | 19.04634 | 29.56362 | 35.06225 | 24.56515 |
|2| 15.78000 | 19.69866 | 29.96626 | 35.01998 | 24.52287 |
|3| 16.87761 | 20.38502 | 30.37017 | 35.63714 | 25.20923 |
|4| 16.69275 | 20.61141 | 30.59656 | 35.93273 | 25.43562 |
|5| 17.73927 | 20.58639 | 31.65695 | 36.58856 | 26.09145 |
|6| 16.12396 | 20.13286 | 30.31022 | 35.36394 | 24.86683 | 
|7| 16.67978 | 19.91760 | 30.58359 | 35.91976 | 25.42266 | 
|8| 16.62444 | 19.84851 | 30.81071 | 37.39098 | 25.36732 | 
|9| 17.23136 | 20.82900 | 30.72391 | 36.06008 | 25.56297 | 
|10| 16.62700 | 19.86482 | 30.53082 |35.86698 | 25.36988 | 

