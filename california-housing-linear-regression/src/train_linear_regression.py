"""Train and evaluate a Linear Regression model on California Housing.

This script is intentionally written as a small, readable baseline project:

1. Load the built-in California Housing dataset from scikit-learn.
2. Split the data into training and test sets.
3. Standardize the input features.
4. Train a Linear Regression model.
5. Evaluate the model with common regression metrics.
6. Save two diagnostic figures:
   - true values vs. predicted values: checks prediction accuracy visually
   - residuals vs. predicted values: checks whether errors show a pattern

The target value in this dataset is median house value measured in units of
100,000 dollars. For example, a target value of 2.5 means about $250,000.
The dataset target is capped near 5.0, so a vertical band at true value 5.0 is
expected in the true-vs-predicted plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Resolve paths from this file instead of from the current terminal directory.
# This makes the script work whether it is run from the project root or from
# another directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the complete regression experiment."""

    # Load the dataset as pandas objects.
    #
    # as_frame=True returns:
    # - data.data as a pandas DataFrame
    # - data.target as a pandas Series
    #
    # This is convenient because feature names are preserved and can be reused
    # later when we inspect model coefficients.
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    # Show a quick preview so the user can verify that the dataset loaded
    # correctly. X contains the input features; y contains the target value.
    print("Features:")
    print(X.head())
    print("\nTarget:")
    print(y.head())

    # Split the dataset into training and testing parts.
    #
    # test_size=0.2 means 20% of the data is reserved for final evaluation.
    # random_state=42 makes the split reproducible, so different runs produce
    # the same train/test split and the same metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Standardize features to zero mean and unit variance.
    #
    # LinearRegression can technically work without scaling, but scaling makes
    # coefficients easier to compare because every feature is measured on the
    # same standardized scale.
    #
    # Important:
    # - fit_transform is used only on training data.
    # - transform is used on test data.
    #
    # This avoids data leakage, because the test set should not influence the
    # mean and standard deviation learned by the scaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train an ordinary least squares Linear Regression model.
    #
    # The model learns one coefficient for each input feature and one intercept.
    # After training, predictions are computed as:
    # prediction = intercept + sum(feature_i * coefficient_i)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict house values for the unseen test data.
    # These predictions are compared with y_test to estimate generalization.
    y_pred = model.predict(X_test_scaled)

    # Evaluate model performance.
    #
    # MAE:
    #   Average absolute prediction error. Easier to interpret because it is in
    #   the same unit as the target.
    #
    # MSE:
    #   Average squared prediction error. Penalizes large errors more strongly.
    #
    # RMSE:
    #   Square root of MSE, also in the same unit as the target.
    #
    # R2:
    #   Proportion of target variance explained by the model. Higher is better;
    #   1.0 is perfect, 0.0 means no better than predicting the mean.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation:")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # Inspect feature coefficients.
    #
    # Because the features were standardized, larger absolute coefficients
    # generally indicate stronger influence in this linear model. This is still
    # model-specific interpretation, not a proof of real-world causality.
    coef_df = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": model.coef_,
        }
    ).sort_values(by="coefficient", key=abs, ascending=False)

    print("\nFeature coefficients:")
    print(coef_df)

    # Basic data and prediction-range checks.
    #
    # These lines help confirm that the full dataset is used and that unusual
    # points in the plots come from real model predictions, not from a plotting
    # or file-saving error.
    capped_targets = (y_test >= 5).sum()
    negative_predictions = (y_pred < 0).sum()
    high_predictions = (y_pred > 5).sum()

    print("\nData check:")
    print(f"Dataset shape: {X.shape}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Test target range     : {y_test.min():.4f} to {y_test.max():.4f}")
    print(f"Prediction range      : {y_pred.min():.4f} to {y_pred.max():.4f}")
    print(f"Targets capped near 5 : {capped_targets}")
    print(f"Predictions below 0   : {negative_predictions}")
    print(f"Predictions above 5   : {high_predictions}")

    # Plot 1: true values vs. predicted values.
    #
    # Why this plot?
    # Regression metrics give useful numbers, but they hide where errors happen.
    # This figure shows whether predictions follow the ideal y = x line.
    #
    # Points close to the diagonal line represent accurate predictions.
    # Points far from the line represent larger prediction errors.
    # The vertical band at True Value around 5.0 is expected because the
    # California Housing target is capped at 5.00001.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, label="Test samples")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("True vs Predicted House Value")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Ideal prediction",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "true_vs_predicted.png", dpi=300)
    plt.close()

    # Residuals are the remaining errors after prediction:
    # residual = true value - predicted value.
    #
    # A good residual plot should be roughly centered around zero without a
    # strong visible pattern. Clear patterns can suggest that a linear model is
    # too simple or that important features/interactions are missing.
    residuals = y_test - y_pred

    # Plot 2: residuals vs. predicted values.
    #
    # Why this plot?
    # This figure checks model bias. If residuals form a clear curve, funnel, or
    # other pattern, the linear model is probably missing nonlinear structure or
    # important interactions. Here, the diagonal upper band is also expected for
    # capped targets because residual = 5.0 - predicted value.
    #
    # The dashed horizontal line marks zero error.
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.4, label="Residuals")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5, label="Zero error")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
