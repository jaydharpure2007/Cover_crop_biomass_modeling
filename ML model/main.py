import warnings
warnings.filterwarnings("ignore")

import time
import os
import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from config import *
from utils import *
from models import *
from tuning import *
from explainability import *
from evaluation import *


def main():
    print("Starting ML workflow")

    # Feature groups used for fusion experiments
    all_lists = [VIs, SFs, SBs, TFs]
    list_names = ["VIs", "SFs", "SBs", "TFs"]

    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # --------------------------------------------------
    # Load dataset (train and test splits)
    # --------------------------------------------------
    Train = pd.read_excel("Data/AGB_train_test.xlsx", sheet_name="train")
    Test = pd.read_excel("Data/AGB_train_test.xlsx", sheet_name="test")

    # --------------------------------------------------
    # Generate feature combinations (feature fusion experiments)
    # --------------------------------------------------
    for r in range(1, 5):
        for idx_combo in combinations(range(4), r):

            start_time = time.time()

            # Select feature groups for current combination
            selected_lists = [all_lists[i] for i in idx_combo]
            selected_names = [list_names[i] for i in idx_combo]

            combined_features = []
            for lst in selected_lists:
                combined_features.extend(lst)

            # Define output folder name based on feature groups
            folder_name = "_".join(selected_names)
            print(f"\nProcessing: {folder_name}")

            # Create output directory for current experiment
            output_dir = os.path.join(OUTPUT_PATH, folder_name)
            os.makedirs(output_dir, exist_ok=True)

            # --------------------------------------------------
            # Feature selection using VIF (remove multicollinearity)
            # --------------------------------------------------
            features_vif, vif_df = vif_feature_selection(
                data=Train,
                features=combined_features,
                target=TARGET,
                vif_threshold=VIF_THRESHOLD
            )

            # Sort VIF values for reporting
            vif_df = vif_df.sort_values(by="VIF", ascending=True)

            print("Remaining features:", features_vif)

            # --------------------------------------------------
            # Prepare training data
            # --------------------------------------------------
            train_data = Train[features_vif + [TARGET]].copy()
            X_train, y_train = preprocess_data(train_data)

            # Standard scaling (fit only on training data)
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            X_train = scaler_x.fit_transform(X_train)
            y_train = scaler_y.fit_transform(y_train)

            X_train = X_train.astype(np.float32)

            # --------------------------------------------------
            # Prepare test data (use same scaler)
            # --------------------------------------------------
            test_data = Test[features_vif + [TARGET]].copy()
            X_test, y_test = preprocess_data(test_data)

            X_test = scaler_x.transform(X_test)
            X_test = X_test.astype(np.float32)

            # --------------------------------------------------
            # Cross-validation setup
            # --------------------------------------------------
            seed = SEED
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

            # --------------------------------------------------
            # Train multiple ML models
            # --------------------------------------------------
            for model_name in MODEL_LIST:

                start_time = time.time()
                print(f"\n================ {model_name} ================")

                # Initialize model
                regressor = get_model(model_name, seed)

                # Define hyperparameter search space
                PARAM_GRIDS = {
                    "RF": RF_PARAM_GRID,
                    "XGB": XGB_PARAM_GRID,
                    "SVR": SVR_PARAM_GRID,
                    "PLSR": PLSR_PARAM_GRID
                }

                param_grid = PARAM_GRIDS[model_name]

                # Special handling for PLSR (depends on feature count)
                if model_name == "PLSR":
                    param_grid = {
                        "n_components": list(range(1, len(features_vif) + 1))
                    }

                # --------------------------------------------------
                # Hyperparameter tuning using RandomizedSearchCV
                # --------------------------------------------------
                _, best_params = tune_model(
                    regressor,
                    param_grid,
                    X_train,
                    y_train,
                    kf,
                    N_ITER_SEARCH,
                    seed
                )

                print("Tuned Hyperparameters:", best_params)

                # Set best parameters to model
                regressor.set_params(**best_params)

                # --------------------------------------------------
                # Cross-validation evaluation
                # --------------------------------------------------
                cv_results = cross_validate_model(
                    regressor=regressor,
                    X_train=X_train,
                    y_train=y_train,
                    kf=kf,
                    scaler_y=scaler_y
                )

                avg_cv_rmse = cv_results["rmse"]
                avg_cv_r2 = cv_results["r2"]
                avg_cv_mae = cv_results["mae"]

                # --------------------------------------------------
                # Train final model on full training data
                # --------------------------------------------------
                regressor.fit(X_train, y_train)

                # SHAP explainability analysis
                df_shap = compute_shap(regressor, model_name, X_train, features_vif)

                # --------------------------------------------------
                # Final evaluation on train and test sets
                # --------------------------------------------------
                results = evaluate_model(
                    regressor=regressor,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    scaler_y=scaler_y
                )

                train_rmse = results["train_rmse"]
                train_r2 = results["train_r2"]
                train_mae = results["train_mae"]

                test_rmse = results["test_rmse"]
                test_r2 = results["test_r2"]
                test_mae = results["test_mae"]

                train_predictions = results["train_pred"]
                test_predictions = results["test_pred"]

                y_train_true = results["y_train_true"]
                y_test_true = results["y_test_true"]

                # Total processing time for model
                total_time = time.time() - start_time

                # --------------------------------------------------
                # Create output tables
                # --------------------------------------------------
                performance_df = pd.DataFrame([
                    {"Dataset": "Training", "RMSE": train_rmse, "R²": train_r2, "MAE": train_mae},
                    {"Dataset": "Cross-Validation", "RMSE": avg_cv_rmse, "R²": avg_cv_r2, "MAE": avg_cv_mae},
                    {"Dataset": "Test", "RMSE": test_rmse, "R²": test_r2, "MAE": test_mae}
                ])

                time_df = pd.DataFrame([{
                    "Processing Time (seconds)": total_time
                }])

                hyperparam_df = pd.DataFrame([best_params])

                train_result_df = pd.DataFrame({
                    "Observed": y_train_true,
                    "Predicted": train_predictions
                })

                test_result_df = pd.DataFrame({
                    "Observed": y_test_true,
                    "Predicted": test_predictions
                })

                # Output file path
                output_file = os.path.join(output_dir, f"{model_name}.xlsx")

                # --------------------------------------------------
                # Save all results to Excel
                # --------------------------------------------------
                with pd.ExcelWriter(output_file) as writer:
                    performance_df.to_excel(writer, sheet_name="Performance", index=False)
                    train_result_df.to_excel(writer, sheet_name="Train Results", index=False)
                    test_result_df.to_excel(writer, sheet_name="Test Results", index=False)
                    time_df.to_excel(writer, sheet_name="Processing Time", index=False)
                    hyperparam_df.to_excel(writer, sheet_name="Best Hyperparameters", index=False)
                    df_shap.to_excel(writer, sheet_name="SHAP", index=False)
                    vif_df.to_excel(writer, sheet_name="VIF", index=False)

                # Final log
                print(performance_df)
                print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
