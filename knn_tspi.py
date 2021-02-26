import numpy as np


class Queue:
    def __init__(self, data:np.ndarray):
        self.__data = data.tolist()

    def enqueue(self, x:float)->None:
        self.__data.append(x)

    def dequeue(self)->float:
        return self.__data.pop(0)

    def as_array(self)->np.ndarray:
        return np.array(self.__data)


class KNeighborsTSPI:
    """K-Nearest Neighbors Time Series Prediction with Invariances

    Parameters
    ----------
    k : int
        Number of neighbors.

    len_query : int
        Subsequence length.

    weights : str
        Weight function using in predicion

    References
    ----------
    .. [1] G. E. A. P. A. Batista, A. R. S. Parmezan,
           ``A Study of the Use of Complexity Measures in the Similarity 
           Search Process Adopted by kNN Algorithm for Time Series Prediction´´,
           2015, Instituto de Ciências Matemáticas e de Computação, 
           Universidade de São Paulo, São Carlos.

    Examples
    --------
    >>> import numpy as np
    >>> from knn_tspi import KNeighborsTSPI
    >>> ts = 0.5*np.arange(60) + np.random.randn(60)
    >>> model = KNeighborsTSPI()
    >>> model.fit(data=ts)
    >>> preds = model.predict(h=5)
    """
    def __init__(self, k:int=3, len_query:int=4, weights:str="uniform"):
        assert k>0, f"Number of neighbors must be at least 1, got {k}"
        assert len_query>2, f"Query length must be at least 3, got {len_query}"
        assert weights in ("uniform", "distance"), \
            f"Weights must be either uniform or distance, got {weights}"
        self.__k = k
        self.__len_query = len_query
        self.__weights = weights
        self.__data = None
        self.__queue = None
        self.__is_fitted = False

    def fit(self, data:np.ndarray)->None:
        """
        Parameters
        ----------
        data : array-like of shape (n_observations,)
               Time series
        """
        try:
            data = data.astype(np.float)
        except:
            raise ValueError(f"Data must be numeric")

        self.__data = data
        self.__queue = Queue(self.__data[-self.__len_query:])
        self.__is_fitted = True

    def predict(self, h:int, g:callable=None)->np.ndarray:
        assert h>0, f"Horizon must be at least 1, got {h}"
        assert self.__is_fitted, "Model hasn't been fitted yet"
        """
        Parameters
        ----------
        h : int
            Forecast horizon

        g : callable
            User defined function that takes an array of distances and
            return its corresponding weights. Default is the inverse of
            distance. Ignored if weights!="distance"

        Returns
        -------
        predictions : array-like of shape (h,)
        """
        predictions = []
        for _ in range(h):
            f = self.__predict(self.__queue.as_array(), g)
            predictions.append(f)
            self.__queue.dequeue()
            self.__queue.enqueue(f)

        return np.array(predictions)

    def __predict(self, query:np.ndarray, g:callable)->float:
        (indexes, min_dists) = self.__similarity_search(query)
        min_k = int(np.sum(np.mod(-indexes, indexes+1)))
        indexes -= 1
        query_mean = np.mean(query)
        query_std = np.std(query, ddof=1)

        if min_k==0:
            raise RuntimeError(f"min_k is zero, this usually means the"\
                +" time series is too short and/or the query length is too big.")

        predictions = []
        for i in range(min_k):
            subseq = self.__data[indexes[i]:indexes[i]+self.__len_query+1]
            subseq = self.__z_score2(subseq, query)
            subseq = subseq*query_std + query_mean
            predictions.append(subseq[self.__len_query])
   
        if self.__weights=="uniform":
            return np.mean(predictions)
        else:
            return self.__weighted_average(min_dists[:min_k], predictions, g)

    def __similarity_search(self, query:np.ndarray)->tuple:
        query = self.__z_score1(query)
        
        min_dists = np.ones(self.__k)*np.float("inf")
        indexes = np.zeros(self.__k)

        for i in range(self.__k):
            for j in range(self.__data.shape[0] - (2*self.__len_query) - 1):
                if not self.__trivial_match(j, indexes, i):
                    subseq = self.__data[j:j+self.__len_query]
                    subseq = self.__z_score1(subseq)
                    d = self.__distance(subseq, query)
                    if np.isnan(d) or np.isinf(d):
                        raise ValueError(f"distance_measure={d}")

                    if d<min_dists[i]:
                        min_dists[i] = d
                        indexes[i] = j+1

        return (indexes.astype(np.int), min_dists)
    
    def __weighted_average(
            self, min_dists:np.ndarray, predictions:list, 
            g:callable)->np.ndarray:
        predictions = np.array(predictions)
        if g is None:
            min_dists = np.where(min_dists==0, 1e-5, min_dists)
            min_dists = 1 / min_dists
        else:
            try:
                min_dists = g(min_dists)
                min_dists = min_dists.reshape(-1)
            except:
                raise RuntimeError(f"Error during weights calculation")
        return np.sum(min_dists*predictions) / np.sum(min_dists)

    def __trivial_match(self, pos:int, indexes:np.ndarray, inc:int)->bool:
        tm = False
        for i in range(inc):
            if np.abs(pos-indexes[i])<=self.__len_query:
                tm = True
                break
        return tm

    def __ED(self, s:np.ndarray, t:np.ndarray)->float:
        return np.sqrt(np.sum((s-t)**2))

    def __CID(self, Q:np.ndarray, C:np.ndarray)->float:
        CE_Q = np.sqrt(np.sum(np.diff(Q)**2)) + np.finfo("float32").eps
        CE_C = np.sqrt(np.sum(np.diff(C)**2)) + np.finfo("float32").eps
        return self.__ED(Q, C) * (max(CE_Q, CE_C)/min(CE_Q, CE_C))

    def __distance(self, t:np.ndarray, s:np.ndarray)->float:
        return self.__CID(t, s)

    def __z_score1(self, s:np.ndarray)->np.ndarray:
        mean, std = np.mean(s), np.std(s)
        return s-mean if std==0 else (s-mean) / std

    def __z_score2(self, s:np.ndarray, q:np.ndarray)->np.ndarray:
        s1 = s[:-1]
        if np.var(s1) > np.var(q):
            m = np.mean(s1)
            d = np.std(s1)
        else:
            m = np.mean(s)
            d = np.std(s)
        return s - m if d==0 else (s - m) / d

