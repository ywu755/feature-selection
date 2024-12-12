# Feature Selection

This repo includes two methods in feature selection: Stepwise and Recursive Feature Elimination (RFE).

### Stepwise regression:
Stepwise regression is to build the regression model from a set of candidate predictor variables
by entering and removing predictors until there is no justifiable reason to enter or remove any more. You can choose from measures such as `AIC`, `BIC`, or `R-squared` to guide the selection of the final model.

### Recursive Feature Elimination (RFE)
RFE is a filter-based feature selection method. It works by searching for a subset of features by starting with all features in the training dataset
and successfully removing features until the desired number remains. For classification problems, you can use `get_classification_models` and `evaluate_classification_model`. For regression problems, use `get_regression_models` and `evaluate_RFE_regression_model`.

RFE requires two key configuration decisions:

Number of features to select: This determines how many features will be retained.
Algorithm used for feature selection: This specifies the model used to evaluate and rank feature importance.
The process involves two main steps:

Determine the optimal number of features to select using RFE.
Explore various algorithms wrapped by RFE to evaluate their performance with the selected features.
