# Feature Selection

This repo includes two methods in feature selection: Stepwise and Recursive Feature Elimination (RFE).

### Stepwise regression:
Stepwise regression is to build the regression model from a set of candidate predictor variables
by entering and removing predictors until there is no justifiable reason to enter or remove any more. You can choose from measures such as `AIC`, `BIC`, or `R-squared` to guide the selection of the final model.

### Recursive Feature Elimination (RFE)
RFE is a filter-based feature selection method. It works by searching for a subset of features by starting with all features in the training dataset
and successfully removing features until the desired number remains. 

For classification problems, you can use `get_classification_models` and `evaluate_classification_model`. For regression problems, use `get_regression_models` and `evaluate_RFE_regression_model`.

RFE requires two key configuration decisions:

- Number of features to select: Find the best number of features to select with RFE.
- Algorithm used for feature selection: Once the best number has been found, explore various algorithms wrapped by RFE.
