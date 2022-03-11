# Packages ----------------------------------------------------------------

library(tidymodels)
library(tidyverse)
library(patchwork)
library(datasets)

# Exploring ---------------------------------------------------------------

data("airquality", package = "datasets")

glimpse(airquality)

tidymodels_prefer(quiet = F)

(df <- airquality %>% 
    tibble() %>% 
    mutate(Month = as_factor(Month),
           Day = as_factor(Day)))

attach(df)

# NAs?
df %>% summarise(across(everything(),
                        ~ sum(is.na(.x))))


# Plots 
theme_set(theme_bw())

{
  (
    ggplot(data = df, aes(x = Solar.R, y = Ozone)) +
      geom_point() +
      stat_smooth(
        method = "lm",
        formula = y ~ x,
        se = F,
        mapping = aes(),
        method.args = list()
      ) -> g1
  )
  (
    ggplot(data = df, aes(x = Wind, y = Ozone)) +
      geom_point() +
      stat_smooth(
        method = "lm",
        formula = y ~ x,
        se = F,
        mapping = aes(),
        method.args = list()
      ) -> g2
  )
  (
    ggplot(data = df, aes(x = Temp, y = Ozone)) +
      geom_point() +
      stat_smooth(
        method = "lm",
        formula = y ~ x,
        se = F,
        mapping = aes(),
        method.args = list()
      ) -> g3
  )
  (
    ggplot(data = df, aes(x = Month, y = Ozone)) + 
      geom_boxplot() -> g4
  )
}

(g4 | g1 /g2 / g3) + plot_annotation(
  title = "Title",
  caption = "caption",
  tag_levels = "I"
)

ggsave(filename = "graphs.png",
       width = 7,
       height = 4)


# Splitting with rsample --------------------------------------------------

# Train-test or resampling technique splits with rsample package

(df_split <- initial_split(
  data = df,
  prop = 3/4,
  strata = Month
))

(df_train <- training(df_split))
(df_test <- testing(df_split))


# Model recipe with recipe ------------------------------------------------

# Pre-process data with recipe package

(aq_recipe <- recipe(
  formula = Ozone ~ Solar.R + Wind + Temp + Month,
  data = df_train
) %>% 
  step_impute_median(Ozone, Solar.R) %>% 
  step_normalize(Solar.R, Wind, Temp))

# What does the pre-process do to the data?
aq_recipe %>% 
  prep() %>% 
  bake(new_data = NULL)

# And to the testing data?
aq_recipe %>% 
  prep() %>% 
  bake(new_data = df_test)


# Specify regression tree model with parsnip -------------------------------------------

# As we did earlier with the linear model, the parsnip package allows us to initialize and specify the desired model along with the engine and mode that need to be applied.

# We will use the rpart package as our engine. We will use 3 tuning parameters; "cost_complexity" to help with tree pruning, "tree_depth" to set the maximum depth of the trees, and "min_n" for determining the number of data points required for a node to split further.

(tree_model <- parsnip::decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>% 
  set_engine("rpart") %>% 
  set_mode("regression"))

# Specify a grid of values for the tuning parameters: allows us to get all possible permutations of the different parameters based on the number of levels

(tree_params <- dials::grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 2
))


# Cross-validation --------------------------------------------------------

set.seed(1)

(cv_folds <- vfold_cv(
  data = df_train,
  v = 5,
  repeats = 3
))


# Fit the regression tree to the cross-validation folds -------------------
# Implement parallel programming. What is happening is that the tuning is taking the first row of the grid parameters and applying it on the different folds on the CV. Once this is over it goes to the next row of the grid parameters and applies the CV again...

set.seed(1)
(model_fit <- tune::tune_grid(
  tree_model,
  aq_recipe,
  resamples = cv_folds,
  grid = tree_params
))


# Model metrics & lot of tuning parameters --------------------------------

# Select the model based on the metrics obtained for each permutation of parameters with tune

collect_metrics(model_fit)

workflowsets::autoplot(model_fit)

# Gives the best model based on RMSE
(best <- tune::select_best(model_fit, "rsq"))

# Finalize the model with tune ------------------------------------------------------

# This is not the fit, this is like saying: "this are the parameters to use when we fit"

(tree_final <- tune::finalize_model(
  x = tree_model,
  parameters = best
))


# Fit final model to the training data with parsnip ------------------------------------

(fit_train <- parsnip::fit(
  object = tree_final,
  Ozone ~ Solar.R + Wind + Temp + Month,
  data = df_train
))


# Fit final model to the testing set with tune -------------------------

(fit_test <- tune::last_fit(
  object = tree_final,
  Ozone ~ Solar.R + Wind + Temp + Month,
  df_split
))

# Collect the metrics

collect_metrics(fit_test)


# Plot of predictions vs true value ---------------------------------------

fit_test %>% 
  collect_predictions() %>% 
  ggplot(aes(x = .pred, y = Ozone)) +
  geom_abline(intercept = 0,
              slope = 1,
              lty = 2, 
              size = 1.2,
              color = "red") +
  geom_point(size = 3)











