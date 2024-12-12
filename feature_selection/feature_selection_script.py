from typing import Dict, List, Tuple, Union

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sklearn
import yaml

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    train_test_split
    )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import statsmodels.api as sm
import statsmodels.formula.api as smf

import der_simulation_service.model.bess_constants as bc
import der_simulation_service.model.adoption.queries as query

MODEL_OPTIONS = {
    "classification": {
        "lr": LogisticRegression,
        "per": Perceptron,
        "cart": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "gbm": GradientBoostingClassifier,
    },
    "regression": {
        "lr": LogisticRegression,
        "cart": DecisionTreeRegressor,
    },
}


dir_path = os.path.dirname(os.path.realpath(__file__)) + "/feature_plots"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
STATS_MODEL = {
    "smf_logit": smf.logit,
    "discrete_Logit": sm.Logit
    }


def get_data_for_rfe(
    feeders: Tuple,
    predictors: List[str],
    target: str,
    der: str,
    dataset: str,
    samples_frac: float,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Function to download a table from bigquery, remove rows with empty
    variables.
    Args:
        feeders: Optionally pass a subset of feeders to collect data records
        predictor: independent predictor column names to query
        target: dependent feature that our method is trying to predict
        der: DER module. Could be bess or ev, but could be extended.
        dataset: IOU table to query. Could be pge, sce or sdge.
        samples_frac: proportion of samples to collect. Could be in the
        range 0 to 1
    Returns:
        A features input pd.DataFrame (X) and target pd.Series (y)
    """
    ders = ["bess", "ev"]
    if der == "bess":
        df = bc.query_training_data(feeders=feeders)
        # filter residential storage adopters without PV systems from training and
        # validation subsets
        df = bc.residential_battery_assumption(df)
    elif der == "ev":
        # EV specific query from queries.py
        df = query.query_premise_training_data(
            adoption_feature_cols=predictors + [target],
            index_col=None,  # hardcoded in query source code to "premise_id"
            dataset=dataset,
            samples_frac=samples_frac,
        )
    else:
        raise KeyError(
            f"{der} is not valid. Must be one of:" f" {', '.join(ders)}"
        )

    # drop rows with NULLs on features
    df = df.dropna()
    # split in sample matrix (X) and target values (y)
    X = df.loc[:, predictors]
    # ensure that int columns are set to base int dtype as this is required
    # by the smf api
    for column_name in X.columns:
        if X[column_name].dtype == "Int64":
            X[column_name] = X[column_name].astype("int")
    y = df.loc[:, target].astype("int")
    return X, y


def encode_categoricals(
    input_df: pd.DataFrame, cat_cols: List[str]
) -> pd.DataFrame:
    """
    Apply OneHotEncoder on a list of categorical columns.
    Returns a dataframe with categorical features encoded.
    """
    ohe = OneHotEncoder(drop="first").fit(input_df[cat_cols])
    for num_elems, (cat_col, categories) in enumerate(
        zip(cat_cols, ohe.categories_)
    ):
        elems = len(categories)

        if elems == 2:
            cat_dict = {c: i for i, c in enumerate(categories)}
            print(
                f"column {cat_col} is binary and has been encoded as {cat_dict}"
            )
        else:
            # note that the first category is dropped
            columns = ohe.categories_[num_elems][1:]
            print(
                f"column {cat_col} has created the following columns: {columns}"
            )
    output = ohe.transform(input_df[cat_cols]).toarray()
    new_column_names = ohe.get_feature_names_out(ohe.feature_names_in_)
    return pd.DataFrame(data=output, columns=new_column_names)


def get_rfe_pipeline(estimator: sklearn.base, num_features: int) -> Pipeline:
    """
    Prepare a single Recursive Feature Elimination pipeline, using the given
    estimator and number of features.

    Args:
        estimator: a classifier or regressor scikit-learn model class
        num_features: how many features you want to select

    Returns:
        an RFE Pipeline ready for fitting to a training set
    """
    rfe = RFE(estimator=estimator(), n_features_to_select=num_features)
    return Pipeline(steps=[("s", rfe), ("m", estimator())])


def collect_rfe_pipelines(
    estimator: sklearn.base, predictors: List[str]
) -> Dict[str, Pipeline]:
    """
    Initialize a collection of Recursive Feature Elimination pipelines
    for each value of num_features from 1 to the total number available.
    Uses a scikit learn base model as the underlying estimator.

    Returns:
        Dict of {num_features: RFE Pipeline}
    """
    return {
        str(num_features): get_rfe_pipeline(
            estimator=estimator, num_features=num_features
        )
        for num_features in range(1, len(predictors) + 1)
    }


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 10,
    n_repeats: int = 3,
    scoring: str = "accuracy",
    random_seed: int = 0,
) -> np.ndarray:
    """
    Evaluate the given model using cross-validation
    Args:
        scoring: metric to perform an evaluation of model performance
            i.e. "accuracy" for classification models,
            "neg_mean_absolute_error" for regression models

    Returns:
        array of cross-validation scores
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed
    )
    return cross_val_score(
        estimator=model,
        X=features,
        y=target,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )


def determine_num_features(
    features: pd.DataFrame,
    target: pd.Series,
    estimator: sklearn.base,
    n_splits: int,
    n_repeats: int,
    scoring: str,
    random_seed: int,
) -> None:
    """
    Evaluate how well a model performs, given various numbers of
    features. Plot results to enable a data scientist to select the ideal
    number of features.
    """
    pipelines = collect_rfe_pipelines(estimator, features.columns)
    results, names = list(), list()
    for name, pipeline in pipelines.items():
        # pipeline_estimator = pipeline.steps[0][1]
        scores = evaluate_model(
            model=pipeline,
            features=features,
            target=target,
            n_splits=n_splits,
            n_repeats=n_repeats,
            scoring=scoring,
            random_seed=random_seed,
        )
        results.append(scores)
        names.append(name)
        print(">%s %.6f (%.6f)" % (name, np.mean(scores), np.std(scores)))
    plt.boxplot(results, labels=names, showmeans=True)
    plt.savefig("feature_plots/determine_num_features.png")
    # plt.show()


def report_rfe_feature_details(
    features: pd.DataFrame,
    target: pd.Series,
    estimator: sklearn.base,
    n_features: int,
) -> None:
    """
    Report which features were selected by RFE, including relative rankings and
    whether or not they were selected. Uses an underlying estimator.
    Args:
        features: input dataframe with all independent features
        target: dependent variable we are trying to predict
        estimator: sklearn model class
        n_features: the number of features which should be selected
    Prints out the report.
    """
    rfe = RFE(estimator=estimator(), n_features_to_select=n_features)
    rfe.fit(features, target)
    for i, feature_name in enumerate(features.columns):
        print(
            "%s, Selected %s, Rank: %.3f"
            % (
                feature_name,
                rfe.support_[i],
                rfe.ranking_[i],
            )
        )


def test_rfe_across_models(
    features: pd.DataFrame,
    target: pd.Series,
    model_options: Dict[str, sklearn.base.BaseEstimator],
    n_features: int,
    n_splits: int,
    n_repeats: int,
    scoring: str,
    random_seed: int,
) -> None:
    """
    Run RFE with a number of different classifiers and the same number of
    features to see if the results are consistent.
    """
    results, names = list(), list()
    for name, estimator in model_options.items():
        pipeline = get_rfe_pipeline(estimator, n_features)
        scores = evaluate_model(
            model=pipeline[0].estimator,
            features=features,
            target=target,
            n_splits=n_splits,
            n_repeats=n_repeats,
            scoring=scoring,
            random_seed=random_seed,
        )
        results.append(scores)
        names.append(name)
        print(">%s %.3f (%.3f)" % (name, np.mean(scores), np.std(scores)))
    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.savefig("feature_plots/test_rfe_across_models.png")
    # plt.show()


def stepwise(
    dataset: pd.DataFrame,
    target: str,
    method: Union[sm.Logit, smf.logit],
    covs: List[str],
    categorical_cols: List[str],
    included: List[str] = [],
    param: str = "aic",
    verbose: bool = False,
) -> List[str]:
    """
    Forward and Backward stepwise Logistic Regression for feature selection

    Args:
        dataset: The training dataset.
        target: The target variable's column name.
        method: formula api from statsmodels package as detailed in this link
            https://www.statsmodels.org/devel/api.html#statsmodels-formula-api
        covs: The list of independent covariates.
        included: The covariates to start the method with, can only contain
            elements in the covs list. Can be empty.
        param: The parameter to base the stepwise regression on.
            Can be one of "aic", "bic", "prsquared".
        verbose: Whether or not to print updates along the way.

    Returns:
        List of covariates selected
    """
    assert set([]).issubset(set(covs))
    steps = pd.DataFrame(index=covs + ["<none>"], columns=["operation", param])
    steps = steps.astype(dtype={"operation": "str", param: "float"})
    excluded = list(set(covs) - set(included))
    while True:
        # Check the metric for the initial model
        fml = f"{target} ~ " + " + ".join(
            included if included else "1"
            )
        steps.at["<none>", param] = getattr(
            method.from_formula(fml, dataset).fit_regularized(
                method='l1',
                disp=False
            ), param
        )
        steps.at["<none>", "operation"] = ""
        if verbose:
            print("{} {: >40}".format(fml, steps.at["<none>", param]))

        # At this step, we go through the features in the list.
        # We start with no features, and create a list to record aic.
        for col in included:
            tmp_included = [x for x in included if x != col]
            fml = f"{target} ~ " + " + ".join(
                [f'C({feature})' if feature in categorical_cols
                else feature for feature in tmp_included]
                if tmp_included else "1"
            )
            steps.at[col, param] = getattr(
                method.from_formula(fml, dataset).fit_regularized(
                    method='l1',
                    disp=False
                ), param
            )
            steps.at[col, "operation"] = "-"
        for col in excluded:
            fml = f"{target} ~ " + " + ".join(
                [f"C({feature})" if feature in categorical_cols
                else feature for feature in included + [col]]
            )
            steps.at[col, param] = getattr(
                method.from_formula(fml, dataset).fit_regularized(
                    method='l1',
                    disp=False
                ), param
            )
            steps.at[col, "operation"] = "+"

        steps = steps.sort_values(by=param)

        # Here we perform stepwise regression with bidirectional elimination.
        # Starting with no features, we add (and potentially delete) features
        # one at a time.
        # At each step we test the impact on AIC of adding (or deleting if
        # already included) each feature. We choose the feature that reduces
        # the AIC the most.
        # When the AIC can no longer be reduced by adding or deleting any of
        # the features, the stepwise regression is complete.

        if verbose:
            print(steps)
        if steps[param].min() < steps.at["<none>", param]:
            min_param = steps.index[steps[param].argmin()]
            if steps.at[min_param, "operation"] == "+":
                included += [min_param]
                if verbose:
                    print(f"***{min_param} added.***")
                excluded.remove(min_param)
            elif steps.at[min_param, "operation"] == "-":
                included.remove(min_param)
                if verbose:
                    print(f"***{min_param} removed.***")
                excluded += [min_param]
        else:
            if verbose:
                final_input_features = " + ".join(
                    [f'C({feature})' if feature in categorical_cols
                        else feature for feature in included]
                        if included else "1"
                        )
                print(
                    f'Final Model: {target} ~ {final_input_features}'
                )
            break
        if verbose:
            print("=================================================")

    return included


def run_all_stepwises(dataset: pd.DataFrame, target: str) -> None:
    """
    Scripty convenience method to run stepwise on `features` with all three
    valid parameter values and the BESS modeling column constants
    """
    stepwise_metrics = ("aic", "bic", "prsquared")
    for stepwise_metric in stepwise_metrics:
        stepwise(
            dataset=dataset,
            target=target,
            method=Union[smf.logit, sm.Logit],
            covs=dataset.columns,
            categorical_cols=[],
            included=[],
            param=stepwise_metric,
            verbose=True,
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml_path",
        type=str,
        metavar="",
        help="path to yaml configuration file. Example of yaml doc located at"
        "reference/bess/features_selection_config.yaml",
    )
    args = parser.parse_args()

    # folder to load config file
    config_path = args.config_yaml_path

    # Function to load yaml configuration file from a path
    def load_config(path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config

    config = load_config(config_path)

    # set feature selection method to use "RFE" or "Stepwise"
    method = config["method"]
    method_name = list(config["method"].keys())[0]

    # set ML supervised models to explore Classifiers or Regressor
    model_options = MODEL_OPTIONS[config["algorithm_type"]]

    # set specific ML model to use
    estimator = model_options[config["estimator"]]

    # evaluate if feeders are given, then convert feeders list in a Tuple
    feeders = config["feeders"]
    if feeders is not None:
        feeders = tuple(feeders)

    # execute pipeline
    X, y = get_data_for_rfe(
        feeders=feeders,
        predictors=config["predictor_cols"],
        target=config["target_col"],
        der=config["der"],
        dataset=config["dataset"],
        samples_frac=config["samples_frac"],
    )
    # identify categorical columns
    cat_cols = config["categorical_cols"]

    # split into training and validation sets stratified by the target column
    if config.get("test_size"):
        test_size = config.get("test_size")
    else:
        test_size = 0.2

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.get("split_random_state"),
        stratify=y,
        )

    if method_name == "rfe":
        rfe_params = config["method"]["rfe"]
        if cat_cols is not None:
            encoded_features = encode_categoricals(X_train, cat_cols)
            # replace encoded categorical features with one-hot-encoding
            X_train.drop(columns=cat_cols, inplace=True)
            X_train = (
                X_train.reset_index()
                .join(encoded_features)
                .set_index("premise_id", drop=True)
            )

        function = rfe_params["function"]
        function_name = list(function.keys())[0]

        if function_name == "determine_num_features":
            params = function["determine_num_features"]
            determine_num_features(
                features=X_train, target=y_train,
                estimator=estimator, **params
            )
        elif function_name == "report_rfe_feature_details":
            params = function["report_rfe_feature_details"]
            report_rfe_feature_details(
                features=X_train, target=y_train,
                estimator=estimator, **params
            )
        elif function_name == "test_rfe_across_models":
            params = function["test_rfe_across_models"]
            test_rfe_across_models(
                features=X_train, target=y_train,
                model_options=model_options, **params
            )
        else:
            print(
                f"{function_name} - invalid function name. "
                "Valid options are: determine_num_features"
                "report_rfe_feature_details and test_rfe_across_models"
            )

    elif method_name == "stepwise":
        dataset = X_train.join(y_train)
        if cat_cols is not None:
            dataset[cat_cols] = dataset[cat_cols].astype("category")
        else:
            cat_cols = []
        stepwise_params = method.get("stepwise")
        stepwise(
            dataset=dataset,
            target=config.get("target_col"),
            method=STATS_MODEL[stepwise_params.get("formula")],
            covs=config["predictor_cols"],
            categorical_cols=cat_cols,
            included=[],
            param=stepwise_params.get("param"),
            verbose=stepwise_params.get("verbose"),
        )
    else:
        print(
            f"{method_name} is not a valid feature "
            "selection method. Valid options are: rfe or stepwise"
        )
