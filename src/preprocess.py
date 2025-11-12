import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

# Assurer que les ressources NLTK sont téléchargées
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

if __name__ == "__main__":
    raw_data_path = os.path.join('data', 'raw_tweets.csv')
    df = pd.read_csv(raw_data_path)
    
    print("Début du prétraitement du texte...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Prétraitement terminé.")
    
    df_processed = df[['sentiment', 'cleaned_text']]
    X = df_processed['cleaned_text']
    y = df_processed['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    data_dir = 'data'
    train_df = pd.DataFrame({'text': X_train, 'sentiment': y_train})
    test_df = pd.DataFrame({'text': X_test, 'sentiment': y_test})
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    
    print("Données d'entraînement et de test sauvegardées.")
