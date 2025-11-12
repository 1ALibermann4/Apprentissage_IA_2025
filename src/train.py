# src/train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import os
from load_data import load_and_prepare_data  # Assurez-vous que load_data.py est dans src/

# Définir le dossier MLflow local
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# Définir le nom de l'expérience
mlflow.set_experiment("Analyse de Sentiments Twitter")

def train_model(model_name, pipeline, X_train, y_train):
    """Entraîne un modèle et log les informations avec MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Logger les paramètres du modèle
        params = pipeline.get_params()
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tfidf_ngram_range", params['tfidf__ngram_range'])
        mlflow.log_param("tfidf_max_features", params['tfidf__max_features'])
        if model_name == 'LogisticRegression':
            mlflow.log_param("clf_max_iter", params['clf__max_iter'])

        # Entraînement
        print(f"Entraînement du modèle {model_name}...")
        pipeline.fit(X_train, y_train)
        print("Entraînement terminé.")

        # Logger le modèle comme artefact
        example = pd.DataFrame({"text": ["Ceci est un exemple de texte."]})
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"{model_name}_pipeline",
            input_example=example
        )
        print(f"Modèle {model_name} loggé avec MLflow.")

if __name__ == "__main__":
    # Charger l'échantillon de données déjà préparé
    data_path = os.path.join('data', 'raw_tweets.csv')
    if not os.path.exists(data_path):
        print("Échantillon de données introuvable, création...")
        dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        df = load_and_prepare_data(dataset_url).sample(n=50000, random_state=42)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    X = df['text'].astype(str)
    y = df['sentiment']

    # Pipeline pour Logistic Regression
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=200))
    ])
    train_model("LogisticRegression", lr_pipeline, X, y)

    # Pipeline pour Naive Bayes
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ])
    train_model("NaiveBayes", nb_pipeline, X, y)
