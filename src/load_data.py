# src/load_data.py
import pandas as pd
import requests
import zipfile
import io
import os

def load_and_prepare_data(url, data_dir='data'):
    """Télécharge et prépare le jeu de données Twitter Sentiment140."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'sentiment140.zip')
    csv_path = os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv')

    # Télécharger le fichier si non présent
    if not os.path.exists(csv_path):
        print("Téléchargement du jeu de données...")
        response = requests.get(url)
        response.raise_for_status()
        # Extraire le contenu du zip en mémoire
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(data_dir)
        print("Téléchargement et extraction terminés.")

    # Définir les colonnes et charger les données
    cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(
        csv_path,
        header=None,
        names=cols,
        encoding='latin-1'
    )

    # Sélectionner les colonnes pertinentes
    df = df[['sentiment', 'text']]
    # Mapper les sentiments : 0 -> 0 (négatif), 4 -> 1 (positif)
    df['sentiment'] = df['sentiment'].replace({4: 1})

    print("Préparation des données terminée.")
    return df

if __name__ == "__main__":
    dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    
    # Charger les données et travailler sur un échantillon de 50 000 lignes pour le TP
    full_df = load_and_prepare_data(dataset_url)
    data_df = full_df.sample(n=50000, random_state=42)

    # Sauvegarder l'échantillon pour les étapes suivantes
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'raw_tweets.csv')
    data_df.to_csv(output_path, index=False)
    print(f"Échantillon de données sauvegardé dans {output_path}")
