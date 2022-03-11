# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)


# Tidy modelling---------------------------------------------------------------

tidymodels_prefer()

# Data download

(nhl <- read_csv("https://raw.githubusercontent.com/thebioengineer/TidyX/master/TidyTuesday_Explained/081-Tidy_Models_5/nhl_data.csv") %>% 
    janitor::clean_names() %>% 
    mutate(playoffs = as.numeric(grepl("\\*", team)) %>% as.factor()))

glimpse(nhl)

attach(nhl)

# Data splitting with rsample----------------------------------------------------------

set.seed(1)

(nhl_split <- initial_split(data = nhl, prop = 3/4, strata = "season"))

glimpse(nhl_split)

# We want a hard split between seasons (train with a season to predict the outcome in another one)

(nhl_season_split <- structure(
  list(
    data = nhl,
    in_id = which(nhl$season == "2019-2020"),
    out_id = NA,
    id = tibble(id = "split")
  ),
  class = c("mc_split", "rsplit")
))

(nhl_train <- training(nhl_season_split))
(nhl_test <- testing(nhl_season_split))



# Resampling: (Repeated) cross-validation with rsample --------------------------------------------

set.seed(1)

(cv_folds <- vfold_cv(
  data = nhl_train,
  v = 5,
  repeats = 2,
  strata = playoffs
))


# Initialize the model with parsnip ---------------------------------------

(glm_model <- logistic_reg() %>% 
   set_engine("glm") %>% 
   set_mode("classification"))


# Set up the recipe with recipes ------------------------------------------

#  No pre-processing with this data set (imputation or normalization)

(nhl_recipe <- recipe(
  formula = playoffs ~ gf_per_g + ga_per_g + sv_pct,
  data = nhl_train
))


# Set up the workflow with workflow ---------------------------------------

(nhl_workflow <- workflow() %>% 
   add_model(glm_model) %>% 
   add_recipe(nhl_recipe))



# Fit the model to cross-validation folds with tune ---------------------------------

(nhl_fit <- nhl_workflow %>% 
   fit_resamples(
     object = glm_model,
     preprocessor = nhl_recipe,
     resamples = cv_folds,
     metrics = metric_set(roc_auc, kap, accuracy)
   ))

nhl_fit %>% collect_metrics()


# Predict on test data ----------------------------------------------------

(fit_nhl <- nhl_workflow %>% 
   last_fit(
     split = nhl_season_split,
     metrics = metric_set(roc_auc, kap, accuracy)
   ))


(nhl_tes_pred <- bind_cols(
  nhl_test,
  fit_nhl %>% collect_predictions() %>% select(starts_with(".pred"))
))


nhl_tes_pred %>% select(season, team, playoffs, .pred_0, .pred_1, .pred_class)

# roc curve

nhl_tes_pred %>% 
  roc_curve(truth = playoffs, .pred_0) %>% 
  autoplot()

fit_nhl %>% 
  collect_metrics()

