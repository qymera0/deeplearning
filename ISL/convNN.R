
# 0 LOAD PACKAGE ----------------------------------------------------------

library(tidyverse)
library(reticulate)
use_condaenv('keras', required = T)
library(keras)
library(jpeg)

# 1 LOAD DATA -------------------------------------------------------------

cifar100 <- dataset_cifar100()

xTrain <- cifar100$train$x

gTrain <- cifar100$train$y

xTest <- cifar100$test$x

gTest <- cifar100$test$y


# 2 WRANGLE DATA ----------------------------------------------------------

xTrain <- xTrain / 255

xTest <- xTest / 255

yTrain <- to_categorical(gTrain, 100)

# Plot some of the figures

par(mar = c(0, 0, 0, 0), mfrow = c(5, 5))

index <- sample(seq(50000), 25)

for (i in index) plot(as.raster(xTrain[i, , , ]))


# 3 MODEL FIT -------------------------------------------------------------


mdlCnn <-
        keras_model_sequential() %>% 
        layer_conv_2d(
                filters = 32,
                kernel_size = c(3, 3),
                # Output channes with same size of inputs
                padding = 'same',
                activation = 'relu',
                input_shape = c(32, 32, 3)
        ) %>% 
        layer_max_pooling_2d(
                pool_size = c(2, 2)
        ) %>% 
        layer_conv_2d(
                filters = 64,
                kernel_size = c(3, 3),
                padding = 'same',
                activation = 'relu'
        ) %>% 
        layer_max_pooling_2d(
                pool_size = c(2, 2)
        ) %>% 
        layer_conv_2d(
                filters = 128,
                kernel_size = c(3, 3),
                padding = 'same',
                activation = 'relu'
        ) %>% 
        layer_max_pooling_2d(
                pool_size = c(2, 2)
        ) %>% 
        layer_conv_2d(
                filters = 256,
                kernel_size = c(3, 3),
                padding = 'same',
                activation = 'relu'
        ) %>% 
        layer_max_pooling_2d(
                pool_size = c(2, 2)
        ) %>% 
        layer_flatten() %>% 
        layer_dropout(rate = 0.5) %>% 
        layer_dense(
                units = 512,
                activation = 'relu'
        ) %>% 
        layer_dense(
                units = 100,
                activation = 'softmax'
        )

mdlCnn %>% 
        compile(
                loss = 'categorical_crossentropy',
                optimizer = optimizer_rmsprop(),
                metrics = c('accuracy')
        )

history <-
        mdlCnn %>% 
        fit(
                xTrain,
                yTrain,
                epocs = 30,
                batch_size = 128,
                validation_split = 0.2
        )
