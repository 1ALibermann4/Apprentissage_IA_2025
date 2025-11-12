import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import os

mlflow.set_experiment("Analyse de Sentiments Twitter")

def train_model(model_name, pipeline):
    with mlflow.start_run(run_name=model_name):
        train_df = pd.read_csv(os.path.join('data', 'train.csv'))
        X_train = train_df['text'].astype(str)
        y_train = train_df['sentiment']

        params = pipeline.get_params()
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tfidf_ngram_range", params['tfidf__ngram_range'])
        mlflow.log_param("tfidf_max_features", params['tfidf__max_features'])
        if model_name == 'LogisticRegression':
            mlflow.log_param("clf_max_iter", params['clf__max_iter'])

        print(f"Entraînement du modèle {model_name}...")
        pipeline.fit(X_train, y_train)
        print("Entraînement terminé.")

        example = pd.DataFrame({"text": ["Ceci est un exemple de texte."]})
        mlflow.sklearn.log_model(pipeline, artifact_path=f"{model_name}_pipeline", input_example=example)

if __name__ == "__main__":
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=200))
    ])
    train_model('LogisticRegression', lr_pipeline)

    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ])
    train_model('NaiveBayes', nb_pipeline)
