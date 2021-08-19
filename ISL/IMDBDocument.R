
# 0 LOAD PACKAGES ---------------------------------------------------------

library(tidyverse)
library(reticulate)
use_condaenv('keras', required = T)
library(keras)
library(Matrix)
library(glmnet)

# 1 LOAD DATA -------------------------------------------------------------

maxFeat <- 10000

imdb <-
        dataset_imdb(
                num_words = maxFeat
        )

# Shortcut to unpacking list of lists

c(c(xTrain, yTrain), c(xTest, yTest)) %<-% imdb


# 2 DATA WRANGLE ----------------------------------------------------------

wordIndex <- dataset_imdb_word_index()

decode_view <- function(text, wordIndex){
        
        word <- names(wordIndex)
        
        idx <- unlist(wordIndex, use.names = F)
        
        word <- c("<PAD >", "<START >", "<UNK >", "<UNUSED >", word)
        
        idx <- c(0:3, idx + 3)
        
        words <- word[match(text, idx, 2)]
        
        paste(words, collapse = ' ')
        
}

decode_view(xTrain[[1]][1:12], wordIndex)

# One hot encoding each document in a list of documents

one_hot <- function(sequences, dimension){
        
        seqLen <- sapply(sequences, length)
        
        n <- length(seqLen)
        
        rowInd <- rep(1:n, seqLen)
        
        colInd <- unlist(sequences)
        
        sparseMatrix(
                i = rowInd,
                j = colInd,
                dims = c(n, dimension)
        )
        
}

xTrain1h <- one_hot(xTrain, 10000)

xTest1h <- one_hot(xTest, 10000)

# Validation set

set.seed(3)

iVal <- sample(seq(along = yTrain), 2000)

# 3 LASSO REGRESSION ------------------------------------------------------

fitLm <-
        glmnet(
                xTrain1h[-iVal, ],
                yTrain[-iVal],
                family = 'binomial',
                standardize = F
        )

classLmv <- 
        predict(
                fitLm, xTrain1h[iVal, ]
        ) > 0
        

# Function to accuracy

accuracy <- function(pred, truth) mean(drop(pred) == drop(truth))

# Accuracy for difference values of lambda

accLmv <-
        apply(
                classLmv,
                2,
                accuracy,
                yTrain[iVal] > 0
        )

# Plot results

par(mar = c(4, 4, 4, 4), mfrow = c(1, 1))

plot(-log(fitLm$lambda), accLmv)


# 4 NEURAL NET ------------------------------------------------------------

mdlNn <-
        keras_model_sequential() %>% 
        layer_dense(
                units = 16,
                activation = 'relu',
                input_shape = c(10000)
        ) %>% 
        layer_dense(
                units = 16,
                activation = 'relu'
        ) %>% 
        layer_dense(
                units = 1,
                activation = 'sigmoid'
        )

mdlNn %>% 
        compile(
                optimizer = 'rmsprop',
                loss = 'binary_crossentropy',
                metrics = c('accuracy')
        )

history <-
        mdlNn %>% 
        fit(
                xTrain1h[-iVal, ],
                yTrain[-iVal],
                epochs = 20,
                batch_size = 512,
                validation_data = list(
                        xTrain1h[iVal, ],
                        yTrain[iVal]
                )
        )
        

# 5 RECURRENT NN ----------------------------------------------------------

wc <-
        sapply(
                xTrain, 
                length
        )

median(wc)        

sum(wc <= 500) / length(wc)        

# RNN requires all documents have the same length. For bigger ones, it will be restricted, for small ones, the beginning will be pad with blanks.     
        
maxLen <- 500        

xTrain <- pad_sequences(xTrain, maxlen = maxLen)        

xTest <- pad_sequences(xTest, maxlen = maxLen)        

# Model

mdlRnn <-
        keras_model_sequential() %>% 
        layer_embedding(
                input_dim = 10000,
                output_dim = 32
        ) %>% 
        layer_lstm(units = 32) %>% 
        layer_dense(units = 1, activation = 'sigmoid')
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
