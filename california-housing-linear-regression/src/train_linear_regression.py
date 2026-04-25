from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    print("Features:")
    print(X.head())
    print("\nTarget:")
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation:")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    coef_df = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": model.coef_,
        }
    ).sort_values(by="coefficient", key=abs, ascending=False)

    print("\nFeature coefficients:")
    print(coef_df)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("True vs Predicted House Value")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "true_vs_predicted.png", dpi=300)
    plt.close()

    residuals = y_test - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
