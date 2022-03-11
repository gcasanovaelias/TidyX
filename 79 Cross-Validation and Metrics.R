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


# Initialize the model woth parsnip ---------------------------------------

# Parsnip package allows to specify the desired model

(lm_model <- linear_reg() %>% 
   set_engine("lm") %>% 
   set_mode("regression"))


# Set the workflow with workflow ------------------------------------------

# Putting the model and the recipe together to construct the workflow

(aq_workflow <- workflow() %>% 
   add_model(lm_model) %>% 
   add_recipe(aq_recipe))


# Cross-validation with rsample --------------------------------------------------------

set.seed(1)

(cv_folds <- vfold_cv(data = df_train,
                      v = 5,
                      repeats = 10))

# Fit the model to the CV folds and check performance with tune and yardstick-----------

(model_fit <- aq_workflow %>% 
   fit_resamples(
     resample = cv_folds,
     metrics = metric_set(rmse, rsq, mae)
   ))

# Obtain the average of each of the (k = 10) fit metrics evaluated on the k fold holdout

model_fit %>% 
  collect_metrics() %>% 
  arrange(.metric)

# Fit the last model to the entire training set and evaluate on the test set

(fit_lm <- last_fit(
  aq_workflow,
  split = df_split,
  metrics = metric_set(rmse, rsq, mae)
))

# Show metrics evaluated on the testing set

fit_lm %>% 
  collect_metrics() %>% 
  arrange(.metric)
