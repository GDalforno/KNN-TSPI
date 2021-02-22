# K-Nearest Neighbors Time Series Prediction with Invariances

Python implementation of the KNN-TSPI, the full description of the algorithm is available at: https://www.researchgate.net/publication/300414605_A_Study_of_the_Use_of_Complexity_Measures_in_the_Similarity_Search_Process_Adopted_by_kNN_Algorithm_for_Time_Series_Prediction

## Parameter description
- k : number of neighbors;
- len_query : number of observations in each query;
- weights : if "uniform" the prediction is given by the mean, if "distance" then a custom function g that takes the distances and outputs its corresponding weights is applied. The default is the inverse of the distance;
- h : forecasting horizon.
 
## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.datasets import load_woolyrnq
from pmdarima.model_selection import RollingForecastCV
from knn_tspi import KNeighborsTSPI


# Loads the time series
data = load_woolyrnq()

# Sets the knn-tspi hyperparameter
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
knn = KNeighborsTSPI(len_query=len_query)
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
plt.fill_between(rng, intervals[0, :], intervals[1, :], color="cornflowerblue", label="80% confidence")
plt.fill_between(rng, intervals[1, :], intervals[3, :], color="lightsteelblue", label="95% confidence")
plt.fill_between(rng, intervals[2, :], intervals[0, :], color="lightsteelblue")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Y")
plt.legend(loc="lower left")
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/56834802/108731205-186fc400-750b-11eb-97ab-32e739096c5a.png)
