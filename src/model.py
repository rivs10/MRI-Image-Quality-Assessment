
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "median_intensity",
    "contrast",
    "entropy",
    "sharpness",
    "edge_density",
    "mean_gradient",
    "p10",
    "p25",
    "p75",
    "p90",
]


def build_models():
    """Create the baseline and Random Forest classifiers."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train both models and return results and fitted models."""
    results = []
    trained_models = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Weighted F1": f1_score(
                y_test, predictions, average="weighted"
            ),
        })

        trained_models[name] = model

    return pd.DataFrame(results), trained_models
