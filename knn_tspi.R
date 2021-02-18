# SIMILARITY-BASED TIME SERIES PREDICTION
# Antonio Rafael Sabino Parmezan
# Advisor: Gustavo E. A. P. A. Batista
# LABIC / ICMC-USP
# 2018

#--------------------------------------------------------------------------
# k-Nearest Neighbor - Time Series Prediction with Invariances (kNN-TSPI)
#--------------------------------------------------------------------------

z.score1 <- function(s) {
    if(sd(s)==0) {
        z <- s-mean(s)
    } else {
        z <- (s-mean(s))/sd(s)
    }
    return(z)
}

z.score2 <- function(s, q) {
    s1 <- s[1:(length(s)-1)]
    
    if(var(s1)>var(q)) {
        m <- mean(s1)
        d <- sd(s1)
    } else {
        m <- mean(s)
        d <- sd(s)
    }
    
    if(d==0) {
        z <- s-m
    } else {
        z <- (s-m)/d
    }
    return(z)
}

#--------------------------------------------------------------------------
ED <- function(s, t) {
	return(sqrt(sum((s-t)^2)))
}

CID <- function(Q, C) {
    CE.Q <- sqrt(sum(diff(Q)^2)) + 1e-7
    CE.C <- sqrt(sum(diff(C)^2)) + 1e-7
    
    return(ED(Q, C)*(max(CE.Q, CE.C)/min(CE.Q, CE.C)))
}

distance <- function(s, t) {
    return(CID(s, t))
}

#--------------------------------------------------------------------------
trivial.match <- function(pos, indexes, inc, len.query) {
    tm <- F
    for(i in 1:(inc)) {
        if(abs(pos-indexes[i])<=len.query) {
            tm <- T
            break
        }
    }
    return(tm)
}

similarity.search <- function(data, query, k, len.query) {
    len.data <- length(data)
    query <- z.score1(query) 
    
    min.dists <- rep(1, k)*Inf
    indexes <- rep(0, k)
    
    for(i in 1:k) {
        for(j in 1:(len.data-2*len.query-1)) {
            if(!trivial.match(j, indexes, i, len.query)) {
                subseq <- data[j:(j+len.query-1)]
                subseq <- z.score1(subseq) 
                d <- distance(subseq, query)
                
                if(is.na(d) || d==Inf) {
                    warning(paste("distance_measure=",d))
                }
                
                if(d<min.dists[i]) {
                    min.dists[i] <- d
                    indexes[i] <- j
                }
            }
        }
    }
    return(list("min.dists"=min.dists, "indexes"=indexes))
}

#--------------------------------------------------------------------------
weighted.average <- function(min.dists, predictions, g) {
    if(is.null(g)) {
        min.dists[min.dists==0] <- 1e-5
        min.dists <- 1/min.dists
    } else {
        min.dists <- tryCatch({
            g(min.dists)
        }, error = function(cond){stop(cond)})
    }
    return(sum(min.dists*predictions)/sum(min.dists))
}

predict <- function(data, query, k, len.query, weights, g) {
    len.data <- length(data)
    
    cont <- similarity.search(data, query, k, len.query)
    indexes <- cont$indexes
    min.dists <- cont$min.dists
    min.k <- sum((-indexes)%%(indexes+1)) 
    
    query_mean <- mean(query)
    query_std <- sd(query)
    
    predictions <- c()
    for(i in 1:min.k) {
        subseq <- data[indexes[i]:(indexes[i]+len.query)]
        subseq <- z.score2(subseq, query)
        subseq <- subseq*query_std + query_mean 
        
        predictions<- c(predictions, subseq[len.query+1])
    }
    
    if(weights=="uniform") {
        prediction <- mean(predictions)
    } else {
        prediction <- weighted.average(min.dists[1:min.k], predictions, g)
    }
}

#--------------------------------------------------------------------------
knn.tspi <- function(
        ts, k = 3, len.query = 4, weights = "uniform", g = NULL, h = 1) {
    if(k<1) {
        stop(paste("k must be greater than 0, got", k))
    }
    if(len.query<3) {
        stop(paste("len.query must be greater than 2, got", len.query))
    }
    if(weights!="uniform" && weigths!="distance") {
        stop(paste("weights must be either uniform or distance, got", weights))
    }
    if(h<1) {
        stop(paste("h must be greater than 0, got", weights))
    }
    z <- c()
    query <- ts[-1:(-length(ts)+len.query)]
    for(i in 1:h) {
        f <- predict(ts, query, k, len.query, weights, g)
        z <- c(z, f)
        query <- c(query[2:len.query], f)
    }
    return(z)
}

