# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)

tidymodels_prefer(quiet = F)


# Importing data ----------------------------------------------------------

{
  map(
    .x = list(
      "https://raw.githubusercontent.com/thebioengineer/TidyX/master/TidyTuesday_Explained/085-Tidy_Models_9/winequality_white.csv",
      "https://raw.githubusercontent.com/thebioengineer/TidyX/master/TidyTuesday_Explained/085-Tidy_Models_9/winequality_red.csv"
    ),
    .f = ~ read_csv(.x)
  ) %>%
    map2(.y = list("white", "red"),
         .f = ~ mutate(.data = .x, wine_type = .y)) %>%
    reduce(bind_rows) %>% 
    janitor::clean_names() %>% 
    assign(x = "raw_wine", envir = .GlobalEnv)
}

glimpse(raw_wine)

# NAs

raw_wine %>% summarize(across(.cols = everything(), .fns = ~ sum(is.na(.x))))

# Plot

raw_wine %>% 
  ggplot(aes(x = quality)) +
  geom_histogram() +
  facet_wrap(~ wine_type)


# Split data with rsample -------------------------------------------------

set.seed(1)

(init_split <- initial_split(
  data = raw_wine,
  prop = 3/4,
  strata = "wine_type"
))

(train <- training(init_split))
(test <- testing(init_split))


# Resampling: (repeated stratified) cross-validation ----------------------

set.seed(1)

(cv_folds <- vfold_cv(
  data = train,
  v = 5,
  repeats = 3,
  strata = "wine_type"
))


# Set up the model recipes with recipe -------------------------------------

# Recipe 1: normalize the predictors
(norm_pred_recipe <- recipe(
  formula = quality ~ .,
  data = train
) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    step_dummy(wine_type, one_hot = TRUE))

# Recipe 2: apply scaling (standardize) to only one variable
(scale_pred_recipe <- recipe(
  formula = quality ~ .,
  data = train
) %>% 
    step_scale(all_numeric_predictors()) %>% 
    step_dummy(wine_type, one_hot = TRUE))


# What does this recipe actually does to the data?

norm_pred_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  glimpse()


# Specifying the model with parsnip ---------------------------------------

# Standard regression, no tuning required
(lm_spec <- linear_reg() %>% 
   set_engine("lm") %>% 
   set_mode("regression"))

# Random Forest regression with tuning parameters
(rf_spec <- rand_forest(
  mtry = tune()
) %>% 
    set_engine("randomForest", importance = T) %>% 
    set_mode("regression"))

# xgboost regression with multiple tuning parameters
(xgb_spec <- boost_tree(
  trees = tune(),
  mtry = tune(),
  tree_depth = tune()
) %>% 
    set_engine("xgboost", importance = T) %>% 
    set_mode("regression"))


# Set up the workflow set with workflowset --------------------------------

(wf_set <- workflow_set(
  preproc = list(norm_pred_recipe, scale_pred_recipe),
  models = list(lm_spec, rf_spec, xgb_spec),
  cross = T
))


# Fit and tune the workflowsets with tune ---------------------------------

parallel::makePSOCKcluster(8) %>% doParallel::registerDoParallel()

(fit_workflows <- wf_set %>% 
   workflow_map(
     seed = 1,
     fn = "tune_grid",
     grid = 10,
     resamples = cv_folds
   ))

parallel::makePSOCKcluster(8) %>% parallel::stopCluster()


# Evaluate model fits -----------------------------------------------------

autoplot(object = fit_workflows)

collect_metrics(x = fit_workflows)

rank_results(x = fit_workflows,
             rank_metric = "rmse",
             # Results with numerically best submodel per workflow
             select_best = T)


# Extract the best workflow from workflowsets -----------------------------------------------

fit_workflows %>%
  rank_results(rank_metric = "rmse",
               select_best = T) %>%
  pluck("wflow_id", 1) %>%
  extract_workflow(x = fit_workflows) %>% 
  assign(x = "wf_best", envir = .GlobalEnv)

# (wf_best <- extract_workflow(x = fit_workflows, id = "recipe_2_rand_forest"))


# Extract the tuning results from workflowsets ----------------------------

(wf_best_tuned <- fit_workflows %>% 
  filter(wflow_id == "recipe_2_rand_forest") %>% 
  pluck("result",1))

collect_metrics(wf_best_tuned)

autoplot(wf_best_tuned)

select_best(wf_best_tuned, "rmse")



# Fit the final model -----------------------------------------------------

(wf_best_final <- finalize_workflow(
  x = wf_best,
  parameters = select_best(wf_best_tuned, "rmse")
))

(wf_best_final_fit <- wf_best_final %>% 
    last_fit(
      split = init_split
    ))



# Extract predictions on test data and evaluate model ---------------------

# Train model on the entire training data set and evaluate on the testing data (that's why we put the initial_split object)

collect_metrics(wf_best_final_fit)

wf_best_final_fit %>% collect_predictions()

wf_best_final_fit %>% augment()


# Variable importance evaluation ------------------------------------------

wf_best_final_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vip(geom = "col")
