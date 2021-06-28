
# LOAD PACKAGES -----------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(visdat)
library(patchwork)
library(AmesHousing)
library(doParallel)

# IMPORT DATA -------------------------------------------------------------

trainData <- read_csv("xgboost/ames/train.csv")

testData <- read_csv("xgboost/ames/test.csv")

# EDA -_------------------------------------------------------------------

vis_dat(trainData)

DataExplorer::plot_intro(trainData)

# Missing data

p <- DataExplorer::plot_missing(trainData, missing_only = T)

cols_to_remove <- 
        p$data %>% 
        filter(Band %in% c("Remove", "Bad")) %>% 
        pull(feature)

# Target

# Histogram

g1 <- 
        ggplot(trainData, aes(x = Sale_Price)) + 
        geom_histogram(aes(y = ..density..), 
                       colour = "black", 
                       fill = "white") +
        geom_density(alpha = .2, fill = "#FF6666") +  
        labs(x = "", y = "")

# Box Plot

g2 <- 
        ggplot(trainData, aes(y = Sale_Price)) +
        geom_boxplot(aes(x = ""), colour = "black", fill = "white") +
        coord_flip() + 
        labs(x = "", y = "")

# qqplot

g3 <- 
        ggplot(trainData, aes(sample = Sale_Price)) + 
        stat_qq() +
        stat_qq_line() + 
        labs(x = "", y = "")

g3 | g1 / g2

# Applying log

g1 <- 
        ggplot(trainData, aes(x = log(SalePrice))) + 
        geom_histogram(aes(y = ..density..), 
                       colour = "black", 
                       fill = "white") +
        geom_density(alpha = .2, fill = "#FF6666") +  
        labs(x = "", y = "")

# Box Plot

g2 <- 
        ggplot(trainData, aes(y = log(SalePrice))) +
        geom_boxplot(aes(x = ""), colour = "black", fill = "white") +
        coord_flip() + 
        labs(x = "", y = "")

# qqplot

g3 <- 
        ggplot(trainData, aes(sample = log(SalePrice))) + 
        stat_qq() +
        stat_qq_line() + 
        labs(x = "", y = "")

g3 | g1 / g2


# DATA TRANSFORM ----------------------------------------------------------

# Data recipe

sPriceRecipe <-
        recipe(trainData, SalePrice ~.) %>% 
        step_rm(Id, Street, Utilities) %>% 
        step_rm(one_of(cols_to_remove)) %>% 
        step_log(all_numeric(), offset = 1) %>% 
        step_normalize(all_numeric(), -all_outcomes()) %>% 
        step_other(all_nominal(), -all_outcomes(), threshold = 0.01) %>% 
        step_novel(all_predictors(), -all_numeric()) %>%
        step_impute_knn(all_predictors()) %>% 
        step_dummy(all_nominal(), -all_outcomes())

# MODEL SPECIFICATION -----------------------------------------------------

sPriceXgbMdl <-
        boost_tree(
                trees = tune(),
                learn_rate = tune(),
                tree_depth = tune(),
                min_n = tune(),
                loss_reduction = tune(),
                sample_size = tune(),
                mtry = tune()
        ) %>% 
        set_mode('regression') %>% 
        set_engine('xgboost', nthread = ncores)

# Workflow

sPriceWflow <-
        workflow() %>% 
        add_recipe(sPriceRecipe) %>% 
        add_model(sPriceXgbMdl)

# Params

xgbParams <-
        parameters(
                trees(), 
                learn_rate(),
                tree_depth(), 
                min_n(), 
                loss_reduction(),
                sample_size = sample_prop(), 
                finalize(mtry(), trainData) 
        )

xgbParams <-
        xgbParams %>% 
        update(trees = trees(c(100, 500)))

# Cross validation definition

set.seed(123456)

sPriceVfold <- vfold_cv(trainData, v = 5, strata = SalePrice)

# MODEL TUNE --------------------------------------------------------------

ncores = 4

xgboostTtune <-
        sPriceWflow %>%
        tune_bayes(
                resamples = sPriceVfold,
                param_info = xgbParams,
                iter = 30, 
                metrics = metric_set(rmse, mape),
                control = control_bayes(no_improve = 10, 
                                        save_pred = T, verbose = T)
        )

autoplot(xgboostTtune)

# CHECK RESULTS -----------------------------------------------------------

sPriceBestMld <- select_best(xgboostTtune, 'rmse')

print(sPriceBestMld)

# FINALIZE THE MODEL ------------------------------------------------------

sPriceFinalMdl <-
        finalize_model(sPriceXgbMdl, sPriceBestMld)

sPriceWflowFinal <-
        sPriceWflow %>% 
        update_model(sPriceFinalMdl)

sPricexgbFit <- 
        fit(sPriceWflowFinal, data = trainData)

# # CHECK FINAL MODEL -----------------------------------------------------

pred <-
        predict(sPricexgbFit, testData) %>% 
        mutate(model = 'XGBoost',
               .pred = exp(.pred))

