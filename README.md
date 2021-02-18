# K-Nearest Neighbors Time Series Prediction with Invariances

Python and R implementation of the KNN-TSPI, the full description of the algorithm is described at: https://www.researchgate.net/publication/300414605_A_Study_of_the_Use_of_Complexity_Measures_in_the_Similarity_Search_Process_Adopted_by_kNN_Algorithm_for_Time_Series_Prediction

## Python example

```python
import numpy as np
from pmdarima.datasets import load_airpassengers
from knn_tspi import KNeighborsTSPI

ts = load_airpassengers()

model = KNeighborsTSPI(k=3, len_query=12, weights="uniform")
model.fit(ts)
z = model.predict(h=12)
```

## R example

```r
source("knn_tspi.R")

z <- knn.tspi(AirPassengers, k = 3, len.query = 12, weights = "uniform", g = NULL, h = 12)
```
![Figure_1](https://user-images.githubusercontent.com/56834802/108384834-dcc2ba80-71e9-11eb-96aa-2e0c95b0a2a2.png)
