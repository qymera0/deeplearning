
# 0 LOAD PACKAGES ---------------------------------------------------------

library(data.table)
library(mlr)
library(stringr)
library(xgboost)
library(caret)


# 1 LOAD DATA -------------------------------------------------------------

setcol <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target")

train <- 
        read.table(
                "xgboost/dataset/adult.data", 
                header = F, 
                sep = ",", 
                col.names = setcol, 
                na.strings = c(" ?"), 
                stringsAsFactors = F
        )


test <- 
        read.table(
                "xgboost/dataset/adult.test", 
                header = F, 
                sep = ",",
                skip = 1,
                col.names = setcol, 
                na.strings = c(" ?"), 
                stringsAsFactors = F
        )

# Convert dataframe to data table

setDT(train)

setDT(test)

# Check missing values

table(is.na(train))

sapply(train, function(x) sum(is.na(x))/length(x))*100

table(is.na(test))

sapply(test, function(x) sum(is.na(x))/length(x))*100

# Quick data cleaning

test[ ,target := substr(target, start = 1, stop = nchar(target) - 1)]

# Removing leading whitespaces - test funcion is from data.table

char_col <- colnames(train)[sapply(test, is.character)]

for (i in char_col) set(train, j = i, value = str_trim(train[[i]], side = "left"))

for (i in char_col) set(test, j = i, value = str_trim(test[[i]], side = "left"))

#set all missing value as "Missing" 

train[is.na(train)] <- "Missing" 

test[is.na(test)] <- "Missing"

# 2 FEATURE ENGINEERING ---------------------------------------------------

# Hot encoding

labels <- factor(train$target)

testLabels <- factor(test$target)

newTr <-
        model.matrix(
                ~.+0,
                data = train[ ,-c('target'), with = F]
        )

newTs <- 
        model.matrix(
                ~.+0,
                data = test[ ,-c('target'), with = F]
        )

setdiff(colnames(newTr), colnames(newTs))

newTr <- newTr[ ,colnames(newTr) != 'native.countryHoland-Netherlands']

# Convert factor to numeric

labels <- as.numeric(labels) - 1

testLabels <- as.numeric(testLabels) - 1


# Convert data to DMatrix

dTrain <- xgb.DMatrix(data = newTr, label = labels)

dTest <- xgb.DMatrix(data = newTs, label = testLabels)

# 3 MODELING --------------------------------------------------------------

## 3.1 Initial model ------------------------------------------------------

# Set models parameters

params <-
        list(
                booster = "gbtree", 
                objective = "binary:logistic", 
                eta = 0.3, 
                gamma = 0, 
                max_depth = 6, 
                min_child_weight = 1, 
                subsample = 1, 
                colsample_bytree = 1,
                eval_metric = 'error'
        )

xgbcv <-
        xgb.cv(
                params = params,
                data = dTrain,
                nrounds = 100,
                nfold = 5,
                showsd = T,
                stratified = T,
                print_every_n = 10,
                early_stopping_rounds = 20,
                maximize = F
        )

# Fit model

xgb1 <-
        xgb.train(
                params = params,
                data = dTrain,
                nrounds = 70,
                watchlist = list(val = dTest, train = dTrain),
                print_every_n = 10,
                early_stopping_rounds = 10,
                maximize = F
        )

# Model prediction

xgbpred <- predict(xgb1, dTest)

xgbpred <- ifelse(xgbpred > 0.5, 1, 0)

# Confusion matrix

confusionMatrix(factor(xgbpred), factor(testLabels))

# Important variables

mat <- xgb.importance(feature_names = colnames(newTr), model = xgb1)

xgb.plot.importance(importance_matrix = mat[1:20])