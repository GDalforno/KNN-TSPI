# K-Nearest Neighbors Time Series Prediction with Invariances

Python and R implementation of the KNN-TSPI, the full description of the algorithm is described at: https://www.researchgate.net/publication/300414605_A_Study_of_the_Use_of_Complexity_Measures_in_the_Similarity_Search_Process_Adopted_by_kNN_Algorithm_for_Time_Series_Prediction

## Python example

```python
import matplotlib.pyplot as plt
import numpy as np
from pmdarima.datasets import load_airpassengers
from knn_tspi import KNeighborsTSPI

ts = load_airpassengers()
h = 12

model = KNeighborsTSPI(k=3, len_query=12, weights="distance")
model.fit(ts)
g = lambda distance : np.exp(-distance**2)
z = model.predict(h=12, g=g)

plt.title("AirPassengers - KNN-TSPI predictions")
plt.plot(ts, label="observations", color="blue")
plt.plot(range(ts.shape[0], ts.shape[0]+h), z, label="predictions", color="red")
plt.legend()
plt.grid()
plt.show()
```
![Figure_1](https://user-images.githubusercontent.com/56834802/108384834-dcc2ba80-71e9-11eb-96aa-2e0c95b0a2a2.png)

## R example

```R
```library(ggplot2)
