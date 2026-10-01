import os
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def execute_pipeline():
    print("MLOps Pipeline - Starting pipeline execution...")

    # Charger les données
    iris = load_iris(as_frame=True)
    df = iris.frame

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Séparer les données
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Entraîner le modèle
    classifier = RandomForestClassifier(
        n_estimators=180,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    # Évaluer le modèle
    accuracy = classifier.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2%}")

    # Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)

    with open("models/iris_model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    print("Model saved successfully.")


if __name__ == "__main__":
    execute_pipeline()