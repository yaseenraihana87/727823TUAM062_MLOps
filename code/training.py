import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import time
import os
import random
import tempfile
import joblib

# Load dataset
df = pd.read_csv("code/data/steel_faults.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode target (important)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("SKCT_727823TUAM062_ProductDefectClassification")

# Run multiple experiments
for i in range(12):
    with mlflow.start_run():

        seed = random.randint(1, 1000)

        model = RandomForestClassifier(
            n_estimators=100 + i * 20,
            max_depth=10,
            random_state=seed
        )

        # Training
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        # Predictions
        preds = model.predict(X_test)

        # For multi-class ROC AUC
        try:
            probs = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, probs, multi_class="ovr")
        except:
            auc = 0.0

        # Metrics
        f1 = f1_score(y_test, preds, average="weighted")
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")

        # Model size
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            joblib.dump(model, tmp.name)
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)

        # Log metrics
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("training_time_seconds", end - start)
        mlflow.log_metric("model_size_mb", size_mb)

        # Log parameters
        mlflow.log_param("n_estimators", 100 + i * 20)
        mlflow.log_param("random_seed", seed)

        # Tags
        mlflow.set_tag("student_name", "Yaseen Raihana")
        mlflow.set_tag("roll_number", "727823TUAM062")
        mlflow.set_tag("dataset", "Steel Plates Faults Dataset")

        # Log model
        mlflow.sklearn.log_model(model, "model")