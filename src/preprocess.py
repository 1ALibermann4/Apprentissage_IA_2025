# src/preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

# Téléchargement des ressources NLTK nécessaires (une seule fois)
for resource, download_name in [
    (stopwords.words('english'), 'stopwords'),
    (nltk.data.find('tokenizers/punkt'), 'punkt'),
    (nltk.data.find('corpora/wordnet'), 'wordnet')
]:
    try:
        _ = resource
    except LookupError:
        nltk.download(download_name)

def preprocess_text(text):
    """Nettoie et prétraite un texte de tweet."""
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return " ".join(lemmatized_tokens)

if __name__ == "__main__":
    raw_data_path = os.path.join('data', 'raw_tweets.csv')
    df = pd.read_csv(raw_data_path)

    print("Début du prétraitement du texte...")
    df = df.dropna(subset=['text', 'sentiment'])
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Prétraitement terminé.")

    df_processed = df[['sentiment', 'cleaned_text']]
    X = df_processed['cleaned_text']
    y = df_processed['sentiment']

    # Vérifier que chaque classe a au moins 2 exemples pour stratify
    class_counts = y.value_counts()
    if (class_counts < 2).any():
        print("⚠️ Certaines classes ont moins de 2 exemples, stratify désactivé.")
        stratify_param = None
    else:
        stratify_param = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    data_dir = 'data'
    train_df = pd.DataFrame({'text': X_train, 'sentiment': y_train})
    test_df = pd.DataFrame({'text': X_test, 'sentiment': y_test})
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    print("Données d'entraînement et de test sauvegardées.")
