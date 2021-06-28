
# 0 LOAD PACKAGES ---------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(readr)
library(vip)

# 1 LOAD DATA -------------------------------------------------------------

url <- 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-19/vb_matches.csv'

vbMatches <- read_csv(url, guess_max = 76000)

# 2 WRANGLE DATA ----------------------------------------------------------

# Organize the data

vbParsed <-
        vbMatches %>% 
        transmute(
                circuit,
                gender,
                year,
                w_attacks = w_p1_tot_attacks + w_p2_tot_attacks,
                w_kills = w_p1_tot_kills + w_p2_tot_kills,
                w_errors = w_p1_tot_errors + w_p2_tot_errors,
                w_aces = w_p1_tot_aces + w_p2_tot_aces,
                w_serve_errors = w_p1_tot_serve_errors + w_p2_tot_serve_errors,
                w_blocks = w_p1_tot_blocks + w_p2_tot_blocks,
                w_digs = w_p1_tot_digs + w_p2_tot_digs,
                l_attacks = l_p1_tot_attacks + l_p2_tot_attacks,
                l_kills = l_p1_tot_kills + l_p2_tot_kills,
                l_errors = l_p1_tot_errors + l_p2_tot_errors,
                l_aces = l_p1_tot_aces + l_p2_tot_aces,
                l_serve_errors = l_p1_tot_serve_errors + l_p2_tot_serve_errors,
                l_blocks = l_p1_tot_blocks + l_p2_tot_blocks,
                l_digs = l_p1_tot_digs + l_p2_tot_digs
        ) %>% 
        na.omit()

# Separate dataframes for winners and losers

winners <-
        vbParsed %>% 
        select(
                circuit, gender, year, w_attacks:w_digs
        ) %>% 
        rename_with(~ str_remove_all(., 'w_'), w_attacks:w_digs) %>% 
        mutate(win = 'win')


losers <-
        vbParsed %>% 
        select(
                circuit, gender, year, l_attacks:l_digs
        ) %>% 
        rename_with(~ str_remove_all(., 'l_'), l_attacks:l_digs) %>% 
        mutate(win = 'lose')

vbDf <-
        bind_rows(winners, losers) %>% 
        mutate_if(is.character, factor)


# 3 EXPLORATORY DATA ANALYSIS ---------------------------------------------

vbDf %>% 
        pivot_longer(attacks:digs, names_to = 'stat', values_to = 'value') %>% 
        ggplot(aes(gender, value, fill = win, color = win)) + 
        geom_boxplot(alpha = 0.4) + 
        facet_wrap(~ stat, scales = 'free_y', nrow = 2) + 
        labs(y = NULL, color = NULL, fill = NULL)

# 4 BUILD THE MODEL -------------------------------------------------------

# Split dataset

set.seed(123)

vbSplit <- initial_split(vbDf, strata = win)

vbTrain <- training(vbSplit)

vbTest <- testing(vbSplit)

# Spec model

xgbSpec <-
        boost_tree(
                trees = 1000,
                tree_depth = tune(),
                min_n = tune(),
                loss_reduction = tune(),
                sample_size = tune(),
                mtry = tune(),
                learn_rate = tune()
        ) %>% 
        set_engine('xgboost') %>% 
        set_mode('classification')

xgbSpec

# Spec grid to tune

xgbGrid <-
        grid_latin_hypercube(
                tree_depth(),
                min_n(),
                loss_reduction(),
                sample_size = sample_prop(),
                finalize(
                        # Mtry depends on the actual predictors
                        mtry(),
                        vbTrain
                        ),
                learn_rate(),
                size = 30
        )

# Create workflow

xgbWf <-
        workflow() %>% 
        add_formula(win ~.) %>% 
        add_model(xgbSpec)

# Create cross validation resamples for model tunning

set.seed(123)

vbFolds <- vfold_cv(vbTrain, strata = win)

vbFolds

# 5 TUNE MODEL ------------------------------------------------------------

doParallel::registerDoParallel()

set.seed(234)

xgbRes <-
        tune_grid(
                xgbWf,
                resamples = vbFolds,
                grid = xgbGrid,
                control = control_grid(save_pred = TRUE)
        )

# 6 EXPLORE THE RESULTS ---------------------------------------------------

collect_metrics(xgbRes)

xgbRes %>% 
        collect_metrics() %>% 
        filter(.metric == 'roc_auc') %>% 
        select(mean, mtry:sample_size) %>% 
        pivot_longer(
                mtry:sample_size,
                values_to = 'value',
                names_to = 'parameter'
        ) %>% 
        ggplot(aes(Value, mean, color = parameter)) + 
        geom_point(alpha = 0.8, show.legend = F) +
        facet_wrap(~ parameter, scales = 'free_x') + 
        labs(x = NULL, y = 'AUC')
        
# Show best parameters

show_best(xgbRes, 'roc_auc')

bestAUC <- select_best(xgbRes, 'roc_auc')

bestAUC

# 7 FINALIZE THE WORKFLOW -------------------------------------------------

finalXgb <-
        finalize_workflow(
                xgbWf,
                bestAUC
        )

finalXgb %>% 
        fit(data = vbTrain) %>% 
        pull_workflow_fit() %>% 
        vip(geom = 'point')

# 8 LAST FIT --------------------------------------------------------------

finalRes <-
        last_fit(finalXgb, vbSplit)

collect_metrics(finalRes)

# ROC Curve

finalRes %>%
        collect_predictions() %>%
        roc_curve(win, .pred_win) %>%
        ggplot(aes(x = 1 - specificity, y = sensitivity)) +
        geom_line(size = 1.5, color = "midnightblue") +
        geom_abline(
                lty = 2, alpha = 0.5,
                color = "gray50",
                size = 1.2
        )
