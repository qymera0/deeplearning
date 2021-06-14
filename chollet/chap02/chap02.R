library(reticulate)
use_condaenv('keras', required = T)
library(keras)

# 2.1 FIRST LOOK TO NEURAL NETWORK ----------------------------------------

## Data load ------------------------------------------------------------

mNist <- dataset_mnist()

trainImages <- mNist$train$x

trainLabels <- mNist$train$y

testImages <- mNist$test$x

testLabels <- mNist$test$y

# Create the network structure

netWork <-
        keras_model_sequential() %>% 
        layer_dense(
                units = 512,
                activation = 'relu',
                input_shape = c(28 * 28)
        ) %>% 
        layer_dense(
                units = 10,
                activation = 'softmax'
        ) # Return an array of 10 probability score (summing to one)

# Compile the network

netWork %>% 
        compile(
                optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = c('accuracy')
        )

# Transform the input data to stay between 0 and 1

trainImages <- array_reshape(trainImages, c(60000, 28 * 28))

trainImages <- trainImages / 255

testImages <- array_reshape(testImages, c(10000, 28 * 28))

testImages <- testImages / 255

# Transform the output data to categorical

trainLabels <- to_categorical(trainLabels)

testLabels <- to_categorical(testLabels)

# Fit the network

netWork %>% fit(trainImages, trainLabels, epochs = 5, batch_size = 128)

# Check model performance

metrics <-
        netWork %>% 
        evaluate(testImages, testLabels)

metrics

# Generate predictions

netWork %>% predict_classes(testImages[1:10, ])


# 2.2 DATA REPRESENTATIONS FOR NEURAL NETWORKS ----------------------------

# Vector

x <- c(12, 3, 6, 14, 10)

str(x)

dim(as.array(x))

# Matrix

x <- matrix(rep(0, 3*5), nrow = 3, ncol = 5)

x

dim(x)

# 3D Tensor

x <- array(rep(0, 2 * 3 * 2), dim = c(2, 3, 2))

str(x)

dim(x)

# MNIST Data

trainImages <- mNist$train$x

trainLabels <- mNist$train$y

testImages <- mNist$test$x

testLabels <- mNist$test$y

# Number of axes

length(dim(trainImages))

# Number of dimensions

dim(trainImages)

# Data types

typeof(trainImages)

# Plot 5th digit

digit <- trainImages[5, , ]

plot(as.raster(digit, max = 255))

## Manipulating tensors in R ---------------------------------------------

# Slicing

mySlice <- trainImages[10:99, , ]

dim(mySlice)

# 2.3 Tensor operation ----------------------------------------------------

## 2.3.1 Element-wise operations ------------------------------------------


