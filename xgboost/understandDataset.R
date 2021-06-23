
# LOAD LIBRARIES ----------------------------------------------------------

library(xgboost)
library(Matrix)
library(data.table)
library(vcd)

# DATA SEET PREPARATION ---------------------------------------------------

## Numeric vs categorical variables ---------------------------------------

## Convert categorical to numerical ---------------------------------------

data("Arthritis")

df <- data.table(Arthritis, keep.rownames = F)

head(df)

str(df)

## Create of a new feature based on old ones ------------------------------

# Grouping per 10 years

head(df[ ,AgeDiscret := as.factor(round(Age / 10, 0))])

# Random split in two groups

head(df[ ,AgeCat := as.factor(ifelse(Age > 30, 'Old', 'Young'))])

# Cleaning data

df[ ,ID := NULL]

levels(df[ ,Treatment])

## One-hot encoding ------------------------------------------------------

sparseMatrix <- sparse.model.matrix(Improved ~. -1, data = df)

head(sparseMatrix)

output_vector = df[ ,Improved] == 'Marked'


# BUILD THE MODEL ---------------------------------------------------------

bst <- 
        xgboost(
                data = sparseMatrix, 
                label = output_vector, 
                max.depth = 4,
                eta = 1, 
                nthread = 2, 
                nrounds = 10,
                objective = "binary:logistic",
                eval_metric = 'error'
        )

# FEATURE IMPORTANCE ------------------------------------------------------

importance <-
        xgb.importance(
                feature_names = sparseMatrix@Dimnames[[2]],
                model = bst
        )

head(importance)

## Improvement in the interpretability ------------------------------------

importanceRaw <-
        xgb.importance(
                feature_names = sparseMatrix@Dimnames[[2]],
                model = bst,
                data = sparseMatrix,
                label = output_vector
        )

importanceClean <- importanceRaw[,`:=`(Cover = NULL, Frequency = NULL)]

head(importanceRaw)


## Ploting feature importance ---------------------------------------------

xgb.plot.importance(importance_matrix = importanceRaw)

## Statistical test -------------------------------------------------------

c2 <- chisq.test(df$Age, output_vector)

c2

c3 <- chisq.test(df$AgeDiscret, output_vector)

c3
