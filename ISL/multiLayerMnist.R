
# 0 LOAD PACKAGES ---------------------------------------------------------

library(tidyverse)
library(reticulate)
use_condaenv('keras', required = T)
library(keras)

# 1 LOAD DATA -------------------------------------------------------------

mnist <- dataset_mnist()

xTrain <- mnist$train$x

gTrain <- mnist$train$y

xTest <- mnist$test$x

gTest <- mnist$test$y

# 2 DATA WRANGLE ----------------------------------------------------------

# One hot encode for class label

xTrain <- array_reshape(xTrain, c(nrow(xTrain), 784))

xTest <- array_reshape(xTest, c(nrow(xTest), 784))

yTrain <- to_categorical(gTrain, 10)

ytest <- to_categorical(gTest, 10)

# Rescale X

xTrain <- xTrain / 255

xTest <- xTest / 255


# 3 FIT NEURAL NETWORK ----------------------------------------------------

mdlNn <-
        keras_model_sequential() %>% 
        # Specify layer 01
        layer_dense(
                units = 256,
                activation = 'relu',
                input_shape = c(784)
        ) %>% 
        layer_dropout(rate = 0.4) %>%
        # Specify layer 02
        layer_dense(units = 128, activation = 'relu') %>% 
        layer_dropout(rate = 0.3) %>%
        # Specify output layer
        layer_dense(units = 10, activation = 'softmax')

summary(mdlNn)

# Model optimizer

mdlNn %>% 
        compile(
                loss = "categorical_crossentropy",
                optimizer = optimizer_rmsprop(),
                metrics = c("accuracy")
        )

history <-
        mdlNn %>% 
        fit(
                xTrain,
                yTrain,
                epochs = 30,
                batch_size = 128,
                validation_split = 0.2
        )

plot(history)

# Functions for accuracy

accuracy <- function(pred, truth){
        
        mean(drop(pred) == drop(truth))
}

mdlNn %>% 
        predict_classes(xTest) %>% 
        accuracy(gTest)
