# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)

tidymodels_prefer()


# Notes -------------------------------------------------------------------

# What do you do if you need to run a ton of models all at once? workflowsets.

# workflow is when we fit 1 model, when we fit multiple models we use workflowset and nest them into a list-column


# Tidy modelling ----------------------------------------------------------

(insurance <- read_csv("https://raw.githubusercontent.com/thebioengineer/TidyX/master/TidyTuesday_Explained/084-Tidy_Models_8/insurance.csv"))

glimpse(insurance)

# NAs?
insurance %>% summarize(across(.cols = everything(), .fns = ~ sum(is.na(.x))))

# Graphic
theme_set(theme_bw())

insurance %>% ggplot(aes(x = log(charges))) + 
  geom_histogram()

# Objective: Predict the cost of insurance (charges) based on 6 features


# Splitting data with rsample ----------------------------------------------

set.seed(1)

(insurance_split <- initial_split(
  data = insurance,
  prop = 3/4,
  strata = "region"
))

(train <- training(insurance_split))
(test <- testing(insurance_split))


# Resampling: (repeated stratified) cross-validation with rsample -------------------------------

set.seed(1)

(cv_folds <- vfold_cv(
  data = train,
  v = 5,
  repeats = 3,
  strata = "region"
))


# Set up the model recipe with recipe -------------------------------------

(insurance_rec <- recipe(
  formula = charges ~ .,
  data = train
) %>% 
  # Creates more columns filled with 1s or 0s
  step_dummy(region, one_hot = T) %>% 
  # It does not change the name
  step_log(charges))

# What does this recipe looks like in our training data?
insurance_rec %>% 
  prep() %>% 
  bake(new_data = NULL)


# Model specification with parsnip ----------------------------------------

# With workflowsets we can run different types of models all at once. In this case we will run a linear regression, Random Forest and K Nearest Neighbors.

(lm_spec <- linear_reg() %>% 
   set_engine("lm") %>% 
   set_mode("regression"))

(rf_spec <- rand_forest() %>% 
    set_engine("randomForest", importance = T) %>% 
    set_mode("regression"))

(knn_spec <- nearest_neighbor(neighbors = 4) %>% 
    set_mode("regression"))


# Set up the workflow set with workflowsets --------------------------------

# Combine the pre-processing recipe and the three models together

(wf_set <- workflow_set(
  preproc = list(insurance_rec),
  models = list(lm_spec, rf_spec, knn_spec)
))


# Fit the models to the workflow ------------------------------------------

(insurance_fit <- workflow_map(
  object = wf_set,
  fn = "fit_resamples",
  resamples = cv_folds,
  seed = 1
))


# Evaluate model fits -----------------------------------------------------

autoplot(insurance_fit)

collect_metrics(insurance_fit)

rank_results(x = insurance_fit, rank_metric = "rmse", select_best = T)


# Extract the best workflow -----------------------------------------------

# Form 1
(wf_final <- extract_workflow(
  x = insurance_fit,
  "recipe_rand_forest"
) %>% 
  fit(train))

# Variable importance
wf_final %>% 
  extract_fit_parsnip() %>% 
  vip::vip(geom = "col")


# Predict on test set -----------------------------------------------------

# (preds <- predict(
#   object = wf_final %>% extract_fit_parsnip(),
#   new_data = insurance_rec %>% prep() %>% bake(new_data = test)
# ))
# 
# (test_final <- bind_cols(test, preds))

(
  test_final <- wf_final %>%
    extract_fit_parsnip() %>%
    augment(new_data = insurance_rec %>% prep() %>% bake(new_data = test)) %>%
    mutate(
      unlog_charges = exp(charges),
      unlog_preds = exp(.pred)
    )
)

# Plot of the predictions
test_final %>%
  ggplot(aes(x = unlog_preds, y = unlog_charges)) +
  geom_point(size = 2,
             alpha = 0.5) +
  geom_smooth(method = "lm",
              color = "red",
              size = 2) +
  geom_abline(
    intercept = 0,
    slope = 1,
    size = 2,
    color = "green",
    linetype = "dashed"
  ) +
  labs(title = "Charges")


# Another way to predict on the test data ---------------------------------

(wf_final_full <- extract_workflow(
  x = insurance_fit,
  "recipe_rand_forest"
) %>% 
  last_fit(split = insurance_split))

# wf_final_full %>% select(id, .metrics) %>% unnest(.metrics)
wf_final_full %>% collect_metrics()
  
# wf_final_full %>% select(id, .predictions) %>% unnest(.predictions)
wf_final_full %>% collect_predictions()
