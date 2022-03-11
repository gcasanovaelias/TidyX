# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(discrim) # part of the tidymodels (w/ parsnip)

tidymodels_prefer(quiet = F)


# Tidy modelling ----------------------------------------------------------

data("penguins", package = "palmerpenguins")

glimpse(penguins)

attach(penguins)

# NAs
penguins %>% summarize(across(.cols = everything(), .fns = ~ sum(is.na(.x))))

# Remove two observations with NAs
(penguins_cleaned <- penguins %>% 
    filter(!is.na(bill_depth_mm)))


# Splitting data with rsample ---------------------------------------------

set.seed(1)

(penguin_split <- initial_split(
  data = penguins_cleaned,
  prop = 4/5,
  strata = "species"
))

(train <- training(penguin_split))
(test <- testing(penguin_split))



# Resample: cross-validation with rsample ---------------------------------

set.seed(1)

(cv_folds <- vfold_cv(
  data = train,
  v = 5,
  repeats = 3,
  strata = "species"
))


# Model specification with parsnip (discrim) ----------------------------------------

(nb_model <- discrim::naive_Bayes() %>% 
   set_mode("classification") %>% 
   set_engine("klaR"))

# Set up model recipe with recipe -----------------------------------------

(penguins_recipe <- recipe(
  formula = species ~ .,
  data = train
) %>% 
  step_impute_knn(
    sex,
    neighbors = 3
  ) %>% 
  update_role(
    year, island,
    new_role = "ID"
  ))

# What are the roles of the variables
summary(penguins_recipe)


# Set up the workflow with workflow ---------------------------------------

(penguins_wf <- workflow() %>% 
   add_recipe(penguins_recipe) %>% 
   add_model(nb_model))


# Fit model on training data ----------------------------------------------

# Fit to CV folds

(nb_fit <- penguins_wf %>% 
   fit_resamples(
     resamples = cv_folds
   ))

collect_metrics(nb_fit)


# Make predictions on test data -------------------------------------------

(nb_final <- penguins_wf %>% 
   last_fit(
     split = penguin_split
   ))

collect_metrics(nb_final)

# (nb_test_pred <- test %>% bind_cols(nb_final %>% collect_predictions() %>% dplyr::select(starts_with(".pred_"))))
nb_final %>% augment()

# Confusion matrix
table("predicted class" = nb_test_pred$.pred_class,
      "observed class" = nb_test_pred$species)

# ROC curve
nb_test_pred %>% 
  roc_curve(truth = species,
            .pred_Adelie, .pred_Chinstrap, .pred_Gentoo) %>% 
  autoplot()

