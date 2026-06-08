# src/models.py

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor


def get_model(model_name, seed):
    """
    Simple model factory to standardize model selection.
    """

    # Random Forest (robust baseline, supports parallel training)
    if model_name == "RF":
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=-1
        )

    # Support Vector Regression (kernel-based model)
    elif model_name == "SVR":
        return SVR()

    # Partial Least Squares Regression (handles multicollinearity well)
    elif model_name == "PLSR":
        return PLSRegression()

    # XGBoost (gradient boosting, strong performance on tabular data)
    elif model_name == "XGB":
        return XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )

    # Safety check for invalid input
    else:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            "Expected one of: RF, SVR, PLSR, XGB"
        )