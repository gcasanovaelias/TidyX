
# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)

tidymodels_prefer(quiet = F)


# Tidy modelling ----------------------------------------------------------

data("penguins", package = "palmerpenguins")

glimpse(penguins)

# NAs?
penguins %>% summarise(across(.cols = everything(), ~ sum(is.na(.x)))) %>% View()

penguins %>% filter(is.na(sex))

# Remove the two observations with NA
(penguins_cleaned <- penguins %>% filter(!is.na(bill_depth_mm)))


# Splitting data with rsample ---------------------------------------------

set.seed(1)

(penguin_split <- initial_split(
  data = penguins_cleaned,
  prop = 3/4,
  strata = "species"
))

(train <- training(penguin_split))
(test <- testing(penguin_split))


# Resampling: cross-validation with rsample -------------------------------

set.seed(1)
(cv_folds <- vfold_cv(
  data = train,
  v = 5,
  repeats = 3
))



# Model specification with parsnip ----------------------------------------

(rf_fit <- rand_forest(
  mtry = tune()
) %>% 
  set_engine("randomForest", importance = T) %>% 
  set_mode("classification"))

(rf_tune_grid <- grid_regular(
  mtry(range = c(1, 5)),
  levels = 5
))


# Model recipe with recipe ------------------------------------------------

(penguins_rec <- recipe(
  formula = species ~ .,
  data = train
) %>% 
  step_impute_knn(
    sex,
    neighbors = 3
  ) %>% 
  # Model doesn't use those variables but keeps them in the data
  update_role(year, island, new_role = "ID"))

# Check which variables are predictors, outcome or other

summary(penguins_rec)

# Check the recipe

penguins_rec %>% 
  prep() %>% 
  bake(new_data = NULL)


# Set workflow with workflow ----------------------------------------------

(penguins_wf <- workflow() %>% 
   add_recipe(penguins_rec) %>% 
   add_model(rf_fit))


# Fit the model on the training data --------------------------------------

set.seed(1)

(tune_rf <- tune_grid(
  object = penguins_wf,
  resamples = cv_folds,
  # grid = 5: the algorith chooses combinations based on the integer value
  grid = rf_tune_grid
))

collect_metrics(x = tune_rf)

autoplot(object = tune_rf)

select_best(x = tune_rf, metric = "roc_auc")


# Finalize the model keeping the workflow! --------------------------------

# This way, the workflow already knows the value of the tuned mtry

(penguins_wf_final <- finalize_workflow(
  x = penguins_wf,
  parameters = select_best(x = tune_rf, metric = "roc_auc")
))


# Fit the final model to the training data --------------------------------

(penguins_fit_final <- penguins_wf_final %>% 
   last_fit(
     split = penguin_split
   ))

penguins_fit_final %>% augment()

# Plot the variables of importance ----------------------------------------

penguins_fit_final %>% 
  extract_fit_parsnip() %>% 
  vip::vip(geom = "col",
           aesthetics = list(
             color = "black",
             fill = "palegreen",
             alpha = 0.5
           )) +
  theme_bw()


# Make predictions on test data -------------------------------------------

(fit_test <- penguins_fit_final %>% 
   collect_predictions())

# Confusion matrix

table(actual = fit_test$species, predicted = fit_test$.pred_class)


# Evaluate model performance ----------------------------------------------

collect_metrics(penguins_fit_final)

# Plot predictions vs observations

fit_test %>% 
  roc_curve(
    truth = species,
    # different classes, different ROC curves
    .pred_Adelie, .pred_Chinstrap, .pred_Gentoo
  ) %>% 
  autoplot()
