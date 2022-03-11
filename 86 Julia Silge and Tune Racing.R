# Packages ----------------------------------------------------------------

library(tidymodels)
library(tidyverse)
library(finetune)

tidymodels_prefer(quiet = F)


# Exploratory Data Analysis  ------------------------------------------------------------

# "You can't understate the value of EDA when it comes to machine learning and modelling" - Julia Silge. All you learn from the EDA process impacts all the decision you take later on, typically in ways that set you up for greater success.

# Exploratory Data Analysis (EDA) refers to the critical process of performing initial investigations on data so as to discover patterns, spot anomalies, test hypothesis and to check assumptions with the help of summary statistics and graphical representations. It is an important step to understand the data first gathering as many insights from it; "making sense of data in hand before getting them dirty with it". 

(train_raw <- read_csv("sliced-s01e09-playoffs-1/train.csv"))

glimpse(train_raw)

attach(train_raw)

# How are home runs distributed in the physical space around home plate?

train_raw %>% 
  ggplot(aes(x = plate_x, y = plate_z, z = is_home_run)) +
  stat_summary_hex(alpha = 0.8, bins = 10) +
  scale_fill_viridis_c(labels = percent) +
  labs(fill = "% home-runs")

# How do launch speed and angle of the ball leaving the bat affects home run percentage?

train_raw %>% 
  ggplot(aes(x = launch_angle, y = launch_speed, z = is_home_run)) +
  stat_summary_hex(alpha = 0.8, bins = 15) +
  scale_fill_viridis_c(labels = percent) + 
  labs(fill = "% home runs")

# How does pacing (number of balls, strikes or the inning) affect home runs?

train_raw %>% 
  mutate(is_home_run = if_else(as.logical(is_home_run), "yes", "no")) %>% 
  select(is_home_run, balls, strikes, inning) %>% 
  pivot_longer(-is_home_run) %>% 
  mutate(name = fct_inorder(name)) %>% 
  ggplot(aes(x = value, y = after_stat(density), fill = is_home_run)) +
  geom_histogram(alpha = 0.5, binwidth = 1, position = "identity") +
  facet_wrap(~ name, scales = "free") +
  labs(fill = "Home run?")



# Spending the data budget ----------------------------------------------

# You have a certain amount of data and we have to decide what are we going to do with it (how to "spend it")

# Splitting data with rsample

set.seed(1)

(bb_split <- train_raw %>% 
    mutate(
      is_home_run = if_else(as.logical(is_home_run), "HR", "no"),
      is_home_run = as_factor(is_home_run)
    ) %>% 
    initial_split(prop = 3/4, strata = is_home_run))

(bb_train <- training(bb_split))
(bb_test <- testing(bb_split))


# Resample: (repeated stratiied) cross-validation with rsample

set.seed(1)

(bb_folds <- vfold_cv(
  data = bb_train,
  v = 5,
  repeats = 2,
  strata = is_home_run
))


# Feature engineering (data pre-processing -----------------------------------------------------

# In tidymodels the feature engineering and data pre-processing concept are wrapped up in the recipe. The ingredients, steps and everything in the recipe is estimated with the training data and later applied to other data.

(bb_rec <- recipe(
  formula = is_home_run ~ launch_angle + launch_speed + plate_x + plate_z + bb_type +
    bearing + pitch_mph + is_pitcher_lefty + is_batter_lefty + inning + balls + strikes + game_date,
  data = bb_train
) %>% 
  step_date(game_date, features = c("week", "month"), keep_original_cols = F) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_impute_median(all_numeric_predictors(), -launch_angle, -launch_speed) %>% 
  # impute with linear regression models with those three variables
  step_impute_linear(launch_angle, launch_speed, impute_with = imp_vars(plate_x, plate_z, pitch_mph)) %>% 
  step_nzv(all_predictors()))

# Does the recipe works?

bb_rec %>% prep() %>% bake(new_data = NULL) %>% glimpse()


# Setting up the model and learner ----------------------------------------

# Specifying the model with parsnip

(xgb_spec <- boost_tree(
  trees = tune(),
  min_n = tune(),
  mtry = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification"))

# Setting up the workflow. Workflows are just convenient containers for carrying around bits of modelling pieces (recipes, model specifications, etc)

(xgb_wf <- workflow(preprocessor = bb_rec, spec = xgb_spec))

# We still have to tune the parameters before we fit the model

# Use racing to tune xgboost ----------------------------------------------

# "finetune" is a package developed by Max Kuhn (tidymodels) and RStudio that enhances the "tune" package by providing more specialized methods for finding reasonable values of model tuning parameters. In this way, fine tune is part of the tidymodels but not directly in the "core". tune_race_anova() can be used to eliminate parameter combinations that are not doing well.

# Using racing methods is a great way to tune through a lot of possible parameter options more quickly.

set.seed(1)

(xgb_rs <- finetune::tune_race_anova(
  object = xgb_wf,
  resamples = bb_folds,
  grid = 15,
  metrics = metric_set(mn_log_loss),
  control = finetune::control_race(verbose_elim = T)
))

# Visualize the race

finetune::plot_race(xgb_rs)

# Show the best

show_best(x = xgb_rs)


# Finalize workflow -------------------------------------------------------

# With "finalize" the workflow we refer to the action of actualizing the hyperparameters that we were tuning and that were with "tune()" in the general workflow with the actual values of the best model (based on a metric).

# last_fit(): after determining the best model it is trained with the training data and evaluated on the testing data set.

(xgb_last <- xgb_wf %>% 
   finalize_workflow(parameters = select_best(x = xgb_rs, metric = "mn_log_loss")) %>% 
   last_fit(split = bb_split))

collect_predictions(xgb_last) %>% 
  mn_log_loss(is_home_run, .pred_HR)

collect_metrics(xgb_last)


# Variable importance score -----------------------------------------------

extract_workflow(xgb_last) %>% 
  extract_fit_parsnip() %>% 
  vip::vip(geom = "point", num_features = 15)

