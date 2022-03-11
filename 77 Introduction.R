# Packages ----------------------------------------------------------------
library(tidymodels)


# Modelling in tidymodels---------------------------------------------------------------

# https://www.youtube.com/watch?v=_yPsbxoeRUk&ab_channel=TidyX

data("iris")
attach(iris)

# (1) Initialize the types of models that we want to run with the parsnip package

lm_mod <- linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm")

class(lm_mod) # linear_reg, model_spec

# (2) Fit the model (also with the parsnip package)

fit_lm <- lm_mod %>% 
  fit(Petal.Length ~ Sepal.Width, data = iris)

class(fit_lm) # _lm, model_fit

# (3) Summarize model

fit_lm %>% 
  pluck("fit") %>% 
  summary()

# This would be equal to the output obtained by applying stats::lm()

# Let's use broom to tidy this up

# (4) Model coefficients and standard errors
fit_lm %>% tidy()

# (5) Model fit statistics
fit_lm %>% glance()

# (6) Append fitted and residual values to the original data frame
fit_lm %>% pluck("fit") %>% augment()


# Making predictions in tidymodels ----------------------------------------

# predict.model_fit (parsnip)
predict(object = fit_lm, 
        new_data = iris,
        # "numeric" is default
        type = "pred_int",
        level = 0.95)

# We obtain the new predictions in the same format as in the augment function

# Combine predictions and predictions intervals

iris_pred <- iris %>%
  select(Petal.Length, Sepal.Width) %>%
  bind_cols(predict(fit_lm,
                    new_data = iris)) %>%
  bind_cols(predict(
    object = fit_lm,
    new_data = iris,
    # "numeric" is default
    type = "pred_int",
    level = 0.95
  )) %>% as_tibble()
