from pathlib import Path

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data/raw" / "geyser.tsv"

    df = pd.read_csv(data_path, sep="\t")

    feature_column = "eruptions"
    target_column = "waiting"

    X = df[[feature_column]]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"Train samples: {len(y_train)}  Test samples: {len(y_test)}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE:  {test_mse:.4f}")

    # Export trained pipeline (scaler + model)
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "linear_regression_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
