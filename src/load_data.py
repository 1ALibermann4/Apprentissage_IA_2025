# src/load_data.py
import pandas as pd
import requests
import zipfile
import io
import os

def load_and_prepare_data(url, data_dir='data', sample_size=50000):
    """
    Télécharge, extrait et prépare les données.
    Retourne un échantillon (ou toutes les données si l'échantillon > population).
    """
    # Crée le répertoire de données s'il n'existe pas
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
    full_df = pd.read_csv(
        csv_path,
        header=None,
        names=cols,
        encoding='latin-1'
    )

    # Sélectionner les colonnes pertinentes
    full_df = full_df[['sentiment', 'text']]

    # Mapper les sentiments : 0 -> 0 (négatif), 4 -> 1 (positif)
    full_df['sentiment'] = full_df['sentiment'].replace({4: 1})

    print("Préparation des données terminée.")

    # Tirer un échantillon, mais ne jamais dépasser la population
    actual_sample_size = min(sample_size, len(full_df))
    data_df = full_df.sample(n=actual_sample_size, random_state=42)
    
    # Sauvegarder l'échantillon pour les étapes suivantes
    output_path = os.path.join(data_dir, 'raw_tweets.csv')
    data_df.to_csv(output_path, index=False)
    print(f"Échantillon de données sauvegardé dans {output_path}")

    return data_df

if __name__ == "__main__":
    # URL directe vers le fichier zip du jeu de données
    dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    
    # Charger et préparer les données
    load_and_prepare_data(dataset_url)
