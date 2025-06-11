import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == "__main__":
    # set URL
    mlflow.set_tracking_uri("http://localhost:5001")
    # Set experiment name
    mlflow.set_experiment("assignment-3-mlflow")

    # Load train set
    train_dataset = pd.read_csv("data/train.csv")

    # Get X and Y
    y: np.ndarray = train_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = train_dataset.drop("target", axis=1).values

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=100)
    clf.fit(X, y)

    joblib.dump(clf, "models/model.joblib")

    with mlflow.start_run(run_name="iris-classification") as parent_run:
        # save parent run ID to file
        with open("data/parent_run_id.txt", "w") as f:
            f.write(parent_run.info.run_id)

        mlflow.log_param("dataset_size", len(train_dataset))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y)))

        # Start nested run for model training
        with mlflow.start_run(
            run_name="logistic-regression-training", nested=True
        ) as train_run:
            # get parameter of model
            param = clf.get_params()

            # log model parameter
            mlflow.log_params(param)
            mlflow.log_param("model_type", "LogisticRegression")
            # Infer model signature
            predictions = clf.predict(X)
            signature = infer_signature(X, predictions)

            # Calculate training accuracy
            train_accuracy = clf.score(X, y)
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.log_metric("mean_absolute_error", mae)

            # Log and register the model
            model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                signature=signature,
                input_example=X[:5],
                registered_model_name="iris-logistic-regression",
                metadata={"algorithm": "LogisticRegression", "dataset": "iris"},
            )

            # Log model artifact using joblib as well (for compatibility)
            joblib.dump(clf, "models/model.joblib")
            mlflow.log_artifact("models/model.joblib")
            mlflow.log_artifact("data/train.csv")

            print(
                f"Model is succesfuly logged with the following URI: "
                f"{model_info.model_uri}"
            )
