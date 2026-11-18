
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score

os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Dataset generation
def create_dataset(n, version=1):
    np.random.seed(42 + version)

    delay = np.random.randint(0, 21, n)
    crowd = np.random.randint(1, 6, n)

    if version == 2:
        delay += np.random.randint(5, 16, n)
        crowd = np.minimum(crowd + 1, 5)

    distance = np.random.randint(50, 501, n)
    speed = np.random.uniform(0.7, 1.5, n)
    luggage = np.random.choice(
        ["Small", "Medium", "Large"], n
    )
    transfer = np.random.randint(5, 31, n)
    lift = np.random.choice([0, 1], n)

    # Feature engineering
    walking = distance / (speed * 60)
    walking *= (1 + (crowd - 1) * 0.15)

    luggage_factor = np.where(
        luggage == "Large", 1.3,
        np.where(luggage == "Medium", 1.1, 1.0)
    )

    walking = walking * luggage_factor
    walking += np.where(lift == 0, 1, 0)

    margin = transfer - delay - walking

    missed = (margin < 0).astype(int)

    return pd.DataFrame({
        "train_delay": delay,
        "distance": distance,
        "walking_speed": speed,
        "crowd_density": crowd,
        "luggage": luggage,
        "transfer_time": transfer,
        "lift": lift,
        "estimated_walking_time": walking,
        "transfer_time_margin": margin,
        "missed_connection": missed
    })


# Version management
v1 = create_dataset(250, 1)
v2 = create_dataset(250, 2)

v1.to_csv("datasets/dataset_v1.csv", index=False)
v2.to_csv("datasets/dataset_v2.csv", index=False)

data = pd.concat([v1, v2], ignore_index=True)
data.to_csv("datasets/feature_store.csv", index=False)

print("Version 1:", len(v1))
print("Version 2:", len(v2))
print("Total records:", len(data))

# Features and target
X = data.drop("missed_connection", axis=1)
y = data["missed_connection"]

categorical = ["luggage"]
numerical = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical),
    ("num", "passthrough", numerical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# Model training
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Railway_Bus_Prediction")

best_model = None
best_recall = -1

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model"
        )

        print("\nModel:", name)
        print("Accuracy:", accuracy)
        print("Recall:", recall)

        if recall > best_recall:
            best_recall = recall
            best_model = pipeline

# Save and register selected model
joblib.dump(best_model, "models/best_model.pkl")

with mlflow.start_run(run_name="Best_Model_Registration"):
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        registered_model_name="RailwayBusModel"
    )

print("\nBest model saved!")
print("Best recall:", best_recall)
print("MLflow model registered!")
