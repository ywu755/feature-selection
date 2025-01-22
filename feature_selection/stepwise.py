from typing import List, Optional, Union
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


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
    Run stepwise on `features` with all three
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
