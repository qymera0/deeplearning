
# 0 LOAD PACKAGE ----------------------------------------------------------

library(tidyverse)
library(reticulate)
use_condaenv('keras', required = T)
library(keras)
library(ISLR2)

# 1 LOAD DATA -------------------------------------------------------------

xData <-
        data.matrix(
                NYSE[, c("DJ_return", "log_volume","log_volatility")]
        )

isTrain <- NYSE[ ,'train']


# 2 DATA WRANGLE ----------------------------------------------------------

# Scale

xData <- scale(xData)

# Create lagged versions

lagm <- function(x, k = 1){
        
        n <- nrow(x)
        
        pad <- matrix(NA, k, ncol(x))
        
        return(
                rbind(pad, x[1:(n - k), ])
        )
        
}

arFrame <-
        data.frame(
                log_volume = xData[ , "log_volume"],
                L1 = lagm(xData, 1), 
                L2 = lagm(xData, 2),
                L3 = lagm(xData, 3), 
                L4 = lagm(xData, 4),
                L5 = lagm(xData, 5)
        )

# Remove missing

arFrame <- arFrame[-(1:5), ]

isTrain <- isTrain[-(1:5)]


# 3 FIT LINEAR AR MODEL ---------------------------------------------------

arFit <-
        lm(
                log_volume ~.,
                data = arFrame[isTrain, ]
        )

arPred <- predict(arFit, arFrame[!isTrain, ])

# R^2 at test data

V0 <-
        var(arFrame[!isTrain, 'log_volume'])

1 - mean((arPred - arFrame[!isTrain , "log_volume"])^2) / V0

# Refit including weekDay

arFramed <-
        data.frame(day = NYSE[-(1:5), "day_of_week"], arFrame)

arFitd <-
        lm(
                log_volume ~.,
                data = arFramed[isTrain, ]
        )

arPredd <- predict(arFitd, arFramed[!isTrain, ])

# R^2 at test data

V0d <- var(arFramed[!isTrain, 'log_volume'])

1 - mean((arPredd - arFrame[!isTrain , "log_volume"])^2) / V0

# 4 FIT RNN ---------------------------------------------------------------

## 4.1 Reshape data -------------------------------------------------------

n <- nrow(arFrame)

xRnn <- data.matrix(arFrame[ ,-1])

xRnn <- array(xRnn, c(n, 3, 5))

xRnn <- xRnn[ , , 5:1]

xRnn <- aperm(xRnn, c(1, 3, 2))

dim(xRnn)

## 4.2 Fit the model -----------------------------------------------------

mdRnn <-
        keras_model_sequential() %>% 
        layer_simple_rnn(
                units = 12,
                input_shape = list(5, 3),
                dropout = 0.1,
                recurrent_dropout = 0.1
        ) %>% 
        layer_dense(units = 1)

m