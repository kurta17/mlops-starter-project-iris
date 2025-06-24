import json
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_model() -> dict[str, Any]:
    """Evaluate the trained model and return metrics.

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load unique classes from the original features file
    classes = pd.read_csv("data/features_iris.csv")["target"].unique().tolist()

    # Load test dataset
    test_dataset = pd.read_csv("data/test.csv")
    y: np.ndarray = test_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = test_dataset.drop("target", axis=1).values

    # Load trained model
    clf = joblib.load("models/model.joblib")

    # Make predictions
    prediction: np.ndarray = clf.predict(X)

    # Calculate metrics
    cm: np.ndarray = confusion_matrix(y, prediction)
    f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")
    accuracy: float = accuracy_score(y_true=y, y_pred=prediction)

    # Set MLflow tracking URI and experiment (same as training)
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("assignment-3-mlflow")

    # Read parent run ID from file
    with open("data/parent_run_id.txt") as f:
        parent_run_id = f.read().strip()

    # Continue with the existing parent run from training using run_id
    with mlflow.start_run(run_id=parent_run_id):
        # Start nested run for evaluation metrics
        with mlflow.start_run(run_name="model-evaluation", nested=True):
            # Log evaluation dataset info
            mlflow.log_param("test_dataset_size", len(test_dataset))
            mlflow.log_param("n_test_features", X.shape[1])

            # Log evaluation metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_f1", f1)

            # Log confusion matrix as artifact
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            cm_df.to_csv("data/confusion_matrix.csv")
            mlflow.log_artifact("data/confusion_matrix.csv", "evaluation_artifacts")

            # Log evaluation results
            mlflow.log_artifact("data/eval.json", "evaluation_artifacts")

            print(f"Test accuracy: {accuracy:.4f}")
            print(f"Test F1: {f1:.4f}")

    return {
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }


if __name__ == "__main__":
    metrics = evaluate_model()

    # Save metrics as JSON
    with open("data/eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
