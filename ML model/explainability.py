# src/explainability.py

import shap
import numpy as np
import pandas as pd


def compute_shap(model, model_name, X_train, feature_names):
    """
    Compute global feature importance using SHAP values.
    """

    # Choose SHAP explainer based on model type
    if model_name in ["RF", "XGB"]:
        # Tree-based models: fast and exact
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

    elif model_name == "SVR":
        # Kernel SHAP: model-agnostic but slower
        explainer = shap.KernelExplainer(
            model.predict,
            X_train[:100]  # background sample for approximation
        )
        shap_values = explainer.shap_values(X_train)

    else:
        # Linear models (e.g., PLSR, linear regression variants)
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_train)

    # Mean absolute SHAP value as global importance
    shap_importance = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": shap_importance
    })

    # Rank features by importance
    shap_df = shap_df.sort_values("Importance", ascending=False)

    return shap_df