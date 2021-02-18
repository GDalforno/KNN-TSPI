# K-Nearest Neighbors Time Series Prediction with Invariances

Python and R implementation of the KNN-TSPI, the full description of the algorithm is described at: https://www.researchgate.net/publication/300414605_A_Study_of_the_Use_of_Complexity_Measures_in_the_Similarity_Search_Process_Adopted_by_kNN_Algorithm_for_Time_Series_Prediction

## Python example

```python
import matplotlib.pyplot as plt
from pmdarima.datasets import load_airpassengers
from knn_tspi import KNeighborsTSPI

ts = load_airpassengers()
model = KNeighborsTSPI(k=3, len_query=4, weights="distance")
model.fit(ts)

g = lambda distance : exp(-distance**2)
z = model.predict(h=6, g=g)
```
