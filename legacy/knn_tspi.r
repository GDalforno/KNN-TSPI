#--------------------------------------------------------------------------
# k-Nearest Neighbor - Time Series Prediction with Invariances (kNN-TSPI)
#--------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------
# data : time series
# k : number of neighbors
# len_query : query length
# target : rule to combine the labels from the nearest neighbors
# h : forecast horizon
# g : custom function that takes an array of distances and outputs its
#  corresponding weigths, ignored if target!="custom"
# pred_interval : whether to calculate the 5%, 20%, 80% and 95% quantiles
#  using bootstrapping, this process can be quite time consuming
#--------------------------------------------------------------------------
# Output
#--------------------------------------------------------------------------
# A single array where the h-th position corresponds to the h-th forecasted
#  horizon if pred_interval is set to False, otherwise a dataframe
#  with h rows and 5 columns (mean, 5%, 20%, 80%, 95%)
#--------------------------------------------------------------------------
# References
#--------------------------------------------------------------------------
# [1] G. E. A. P. A. Batista, A. R. S. Parmezan,
#  ``A Study of the Use of Complexity Measures in the Similarity
#  Search Process Adopted by kNN Algorithm for Time Series Prediction´´,
#  2015, Instituto de Ciências Matemáticas e de Computação,
#  Universidade de São Paulo, São Carlos.
#--------------------------------------------------------------------------
knn.tspi <- function(
        data,
        k = 3,
        len_query = 4,
        target = "mean",
        h = 1,
        g = NULL,
        pred_interval = F) {
    if(pred_interval==F) {
        return(predict_global(data, k, len_query, target, h, g))
    } else {
        min_length <- as.integer(3*len_query)
        res <- c()
        for(i in min_length:(length(data)-1)) {
            y <- predict_global(data[1:i], k, len_query, target, g, h = 1)
            res <- c(res, data[i+1]-y)
        }

        mean <- predict_global(data, k, len_query, target, g, h = h)

        futures <- matrix(0L, nrow=1000, ncol=h)
        for(i in 1:1000) {
            futures[i,] <- mean+sample(res, h)
        }
        quants <- t(apply(futures, 2, quantile, probs = c(0.05, 0.2, 0.8, 0.95)))
        quants <- cbind(quants, mean)
        colnames(quants) <- c("Low 95", "Low 80", "High 80", "High 95", "Mean")
        return(quants)
    }
}

predict_global <- function(
        data,
        k = 3,
        len_query = 4,
        target = "mean",
        h = 1,
        g = NULL) {
    data <- as.numeric(data)
    if(any(is.na(data))) {
        warning("Data must be numeric")
    }

    k <- as.integer(k)
    if(k<1) {
        warning(paste("k must be greater than 1, got", k))
    }

    len_query <- as.integer(len_query)
    if(len_query<3) {
        warning(paste("len_query must be greater than 3, got", len_query))
    }

    if(!(target %in% c("mean", "median", "custom"))) {
        warning(
            paste(
                "target must be either ``mean'', ``median'' or ``custom'', got",
                target
            )
        )
    }

    h <- as.integer(h)
    if(h<1) {
        warning(paste("h must be greater than 0, got", k))
    }

    predictions <- rep(0, h)
    query <- tail(data, len_query)
    for(i in 1:h) {
        predictions[i] <- predict_(data, query, k, len_query, target, g)
        query <- c(query[2:len_query], predictions[i])
    }
    return(predictions)
}

#--------------------------------------------------------------------------
predict_ <- function(data, query, k, len_query, target, g) {
    container <- similarity_search(data, query, k, len_query)
    indexes <- container$indexes
    min_dists <- container$min_dists
    min_k <- sum((-indexes)%%(indexes + 1)) # Important

    if(min_k==0) {
        warning("min_k is 0, this usually mean the data is to short and/or the len_query is too large")
    }
    predictions <- rep(0, min_k)

    query_mean <- mean(query)
    query_std <- sd(query)

    for(i in 1 : min_k) {
        subseq <- data[indexes[i] : (indexes[i] + len_query)]
        subseq <- z_score2(subseq, query)
        subseq <- subseq * query_std + query_mean # Inverse function

        # MVA or WA
        predictions[i] <- subseq[(len_query + 1)]
    }

    res <- switch(target,
        "mean"=mean(predictions),
        "median"=median(predictions),
        "custom"=weighted_average(min_dists, predictions, g)
    )
    return(res)
}

#--------------------------------------------------------------------------
similarity_search <- function(data, query, k, len_query) {
    query <- z_score1(query) # Normalization
    min_dists <- rep(1, k) * Inf
    indexes <- rep(0, k)
    for(i in 1 : k) {
        for(j in 1 : (length(data) - (2 * len_query) - 1)) {
            if(!trivial_match(j, indexes, i, len_query)) {
                subseq <- data[j : (j + len_query - 1)]
                subseq <- z_score1(subseq) # Normalization
                d <- distance(subseq, query)
                if (is.na(d) | d == Inf) {
                    warning(paste('distance_measure =', d))
                }

                if (d < min_dists[i]) {
                    min_dists[i] <- d
                    indexes[i] <- j
                }
            }
        }
    }
    return(list("indexes"=indexes, "min_dists"=min_dists))
}

#--------------------------------------------------------------------------
weighted_average <- function(min_dists, predictions, g) {
    if(is.null(g)) {
        min_dists[min_dists==0] <- .Machine$double.eps
        min_dists <- 1 / min_dists
    } else {
        tryCatch(
            expr = {
                min_dists <- g(min_dists)
            },
            error = function(e){
                warning("Error while calculating the weights")
            })
    }
    return(sum(min_dists * predictions) / sum(min_dists))
}
#--------------------------------------------------------------------------
z_score1 <- function(s) {
    if(sd(s) == 0) {
        z <- s - mean(s)
    } else {
        z <- (s - mean(s)) / sd(s)
    }
    return(z)
}

z_score2 <- function(s, q) {
    s1 <- tail(s, -1)
    if(var(s1) > var(q)) {
        m <- mean(s1)
        d <- sd(s1)
    } else {
        m <- mean(s)
        d <- sd(s)
    }

    if(d == 0) {
        z <- s - m
    } else {
        z <- (s - m) / d
    }
    return(z)
}
#--------------------------------------------------------------------------
trivial_match <- function(pos, indexes, inc, len_query) {
    tm <- F
    if((inc-1)<=1) {
        return(tm)
    }
    for(i in 1:(inc-1)) {
        if(abs(pos-indexes[i]) <= len_query) {
            tm <- T
            break
        }
    }
    return(tm)
}
#--------------------------------------------------------------------------
ED <- function(s, t) {
    return(sqrt(sum((s-t)^2)))
}

CID <- function(Q, C) {
    CE_Q <- sqrt(sum(diff(Q)^2)) + .Machine$double.eps
    CE_C <- sqrt(sum(diff(C)^2)) + .Machine$double.eps
    return(ED(Q, C) * (max(CE_Q, CE_C) / min(CE_Q, CE_C)))
}

distance <- function(s, t) {
    return(CID(s, t))
}
