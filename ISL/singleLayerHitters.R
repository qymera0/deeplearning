
# 0 LOAD PACKAGES ---------------------------------------------------------

library(ISLR2)
library(glmnet)
library(reticulate)
use_condaenv('keras', required = T)
library(keras)
library(tidyverse)

# 1 LOAD DATA -------------------------------------------------------------

gitters <- na.omit(Hitters)

n <- nrow(gitters)

set.seed(13)

# Splid data in test and fit

nTest <- trunc(n / 3)

testID <- sample(1:n, nTest)

# 2 FIT LINEAR MODEL ------------------------------------------------------

lFit <-
        lm(
                Salary ~ ., data = gitters[-testID, ]
        )

lPred <- predict(lFit, gitters[testID, ])

with(gitters[testID, ], mean(abs(lPred - Salary)))


# 3 FIT LASSO MODEL -------------------------------------------------------

## 3.1 Scale model matrix ------------------------------------------------

x <- scale(model.matrix(Salary ~. -1, data = gitters))

y <- gitters$Salary


# # 3.2 Fit the model -----------------------------------------------------

cvFit <-
        cv.glmnet(
                x[-testID, ],
                y[-testID],
                type.measure = 'mae'
        )

cPred <- predict(cvFit, x[testID, ], s = 'lambda.min')

mean(abs(y[testID] - cPred))

# # 4 FIT KERAS MODEL -----------------------------------------------------

# Specify the model

modNn <-
        keras_model_sequential() %>% 
        layer_dense(
                units = 50,
                activation = 'relu',
                input_shape = ncol(x)
        ) %>% 
        layer_dropout(rate = 0.4) %>% 
        layer_dense(units = 1)

# Specity the optimizer

modNn %>%
        compile(
                loss = 'mse',
                optimizer = optimizer_rmsprop(),
                metrics = list('mean_absolute_error')
        )

# Train the model

history <-
        modNn %>% 
        fit(
                x[-testID, ],
                y[-testID],
                epochs = 1500,
                batch_size = 32,
                validation_data = list(x[testID, ], y[testID])
        )

plot(history)

# Predict

nPred <- predict(modNn, x[testID, ])

mean(abs(y[testID] - nPred))
