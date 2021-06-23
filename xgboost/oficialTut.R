
# LOAD LIBRARIES ----------------------------------------------------------

library(xgboost)

# DATASET LOADING ---------------------------------------------------------

data(agaricus.train, package = 'xgboost')

data(agaricus.test, package = 'xgboost')

train <- agaricus.train

test <- agaricus.test

dim(train$data)

dim(test$data)


# TRAINING ----------------------------------------------------------------


## Basic training ---------------------------------------------------------

bstSparse <-
        xgboost(
                data = train$data,
                label = train$label,
                max.depth = 2,
                nthread = 2,
                nrounds = 2,
                objective = 'binary:logistic',
                eval_metric = 'error'
        )

## Parameters variations --------------------------------------------------

bstDense <-
        xgboost(
                data = as.matrix(train$data),
                label = train$label,
                max.depth = 2,
                eta = 1,
                nthread = 2,
                nrounds = 2,
                objective = 'binary:logistic',
                eval_metric = 'error'
        )


## xgb.DMatrix ------------------------------------------------------------

dtrain <- xgb.DMatrix(data = train$data, label = train$label)

bstDMatrix <-
        xgboost(
                data = dtrain,
                max.depth = 2,
                nthread = 2,
                nrounds = 2,
                objective = 'binary:logistic',
                eval_metric = 'error'
        )


## Verbose option ---------------------------------------------------------

bst <-
        xgboost(
                data = dtrain,
                max.depth = 2,
                nthread = 2,
                nrounds = 2,
                objective = 'binary:logistic',
                eval_metric = 'error',
                verbose = 2
        )


# BASIC PREDICTION --------------------------------------------------------

## Perform the prediction -------------------------------------------------

# Predicts a probability

pred <- predict(bst, test$data) 

# Needs a threshold


threshold <- 0.5

prediction <- as.numeric(pred > threshold)

## MODEL PERFORMANCE ------------------------------------------------------

err <- mean(as.numeric(pred > threshold) != test$label)

print(paste('test-error=', err))

# ADVANCED FEATURES --------------------------------------------------------

## Data set preparation ----------------------------------------------------

dtrain <- xgb.DMatrix(data = train$data, label = train$label)

dtest <- xgb.DMatrix(data = test$data, label = test$label)


## Learning progress ------------------------------------------------------

watchlist <- list(train = dtrain, test = dtest)

bst <-
        xgb.train(
                data = dtrain,
                max.depth = 2,
                eta = 1,
                nthread = 2,
                nrounds = 2,
                watchlist = watchlist,
                objective = 'binary:logistic',
                eval_metric = 'error',
                eval_metric = 'logloss',
                verbose = 2
                
        )

# # Linear boosting -------------------------------------------------------

bst <-
        xgb.train(
                data = dtrain,
                booster = 'gblinear',
                nthread = 2,
                nrounds = 2,
                watchlist = watchlist,
                objective = 'binary:logistic',
                eval_metric = 'error',
                eval_metric = 'logloss',
                verbose = 2
                
        )

## Manipulating xgb.DMatrix -----------------------------------------------

xgb.DMatrix.save(dtrain, 'dtrain.buffer')

dtrain2 <- xgb.DMatrix('dtrain.buffer')

## Feature importance ----------------------------------------------------

importanceMatrix <- xgb.importance(model = bst)

print(importanceMatrix)

xgb.plot.importance(importance_matrix = importanceMatrix)


## View the trees from a model --------------------------------------------

xgb.dump(bst, with_stats = T)

xgb.plot.tree(model = bst)


## Save and load models ---------------------------------------------------

xgb.save(bst, 'xgboost.model')

# Save as binary vector

rawVec <- xgb.save.raw(bst)

