# Packages ----------------------------------------------------------------

library(tidymodels)
library(tidyverse)
library(datasets)
library(patchwork)

library(lubridate)

tidymodels_prefer()


# patchwork ---------------------------------------------------------------

# https://ggplot2-book.org/arranging-plots.html

# https://patchwork.data-imaginist.com/

# https://cran.r-project.org/web/packages/patchwork/vignettes/patchwork.html


# Exploring ---------------------------------------------------------------

data("airquality", package = "datasets")

glimpse(airquality)

df <- airquality %>% 
  tibble() %>% 
  mutate(Month = as_factor(Month),
         Day = as_factor(Day))

attach(df)

# Are there missing data (NA) in the dataset?
df %>% summarize(across(everything(),
                        ~ sum(is.na(.x))))

# Plots with ggplot2 and patchwork
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


# Modelling ---------------------------------------------------------------

# The main packages in the tidymodels suite that we will use:

# rsample: train/tes or cross validation splits
# recipes: pre- processing of data
# parsnip: specifying the model we want to use
# workflows: putting the model together and constructing the workflow
# yardstick: model evaluation metrics
# broom: model outputs in a clean/tidy data frame

# Splitting data with rsample ---------------------------------------------

(df_split <-
   rsample::initial_split(
     data = df,
     prop = 3 / 4,
     # random sampling is conducted within the stratification variable (resamples have equivalent proportions as the original dataset)
     strata = Month
   ))

(df_train <- rsample::training(df_split))
(df_test <- rsample::testing(df_split))



# Create a model recipe with recipe ---------------------------------------------------

# "setting the recipe" means specifying the model, no different from what we were doing in a normal lm. There are a number of helper functions (step_normalize, step_impute_median, step_dummy) that can be useful for handling data preparation, You would do all pre-processing in this step.

# With tidymodels this step in modelling (that was always done separately) is now incorporated in the workflow

# What to do with the missing values (NAs)? We can remove those observations or make an imputation. step_impute_median is a specification step that will substitute missing values of numeric values of numeric variables by the training set median of those variables 

(
  aq_recipe <- recipes::recipe(formula = Ozone ~ Solar.R + Wind + Temp + Month,
                               data = df_train) %>%
    # substitute NAs for median values
    step_impute_median(Ozone, Solar.R) %>%
    # Normalize variables (mean:0 sd:1)
    step_normalize(Solar.R, Wind, Temp)
)

# To see the pre-process steps applied to the data, you need to prep the recipe and then bake it
aq_recipe %>% 
  prep() %>% 
  # NULL: do it with train_data
  bake(new_data = NULL) # Dependent variable (Ozone) is not normalized, which is good

# If you want to look at the pre-processing on the testing data
aq_recipe %>% 
  prep() %>% 
  bake(new_data = df_test)

# ... The result is not a prediction, what this does it shows us hoe the steps that we have indicated modify the data

# Initialize the model with parsnip ----------------------------------------------------

(
  lm_model <- parsnip::linear_reg() %>%
    parsnip::set_engine("lm") %>%
    parsnip::set_mode("regression")
)
  

# Set up the workflow with workflows -----------------------------------------------------
# This step is initiated with the workflow() function. In the workflow you define "what you expect to happen".

(
  aq_workflow <- workflows::workflow() %>%
    workflows::add_model(lm_model) %>%
    workflows::add_recipe(aq_recipe)
)

# You get an encapsulation of all the steps and data manipulation that you are doing

# Run the model on the training data set with parsnip ----------------------------------

(model_fit <- aq_workflow %>% 
   parsnip::fit(data = df_train))


# Review model ------------------------------------------------------------

model_fit %>% extract_fit_parsnip()

model_fit %>% tidy()

model_fit %>% glance()


# Run model on test set ---------------------------------------------------

(ozone_pred <- predict(
  # model fit
  model_fit %>% extract_fit_parsnip(),
  # bake: apply the pre-processing to new data
  model_fit %>% extract_recipe() %>% recipes::bake(df_test)
))

# (test_baked <- model_fit %>% extract_recipe() %>% recipes::bake(df_test))
# 
# model_fit %>% extract_fit_parsnip() %>% augment(df_test)


