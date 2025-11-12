import pandas as pd
import requests
import zipfile
import io
import os

def load_and_prepare_data(url, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'sentiment140.zip')
    csv_path = os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv')

    if not os.path.exists(csv_path):
        print("Téléchargement du jeu de données...")
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(data_dir)
        print("Téléchargement et extraction terminés.")

    cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(csv_path, header=None, names=cols, encoding='latin-1')
    df = df[['sentiment', 'text']]
    df['sentiment'] = df['sentiment'].replace({4: 1})
    print("Préparation des données terminée.")
    return df

if __name__ == "__main__":
    dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    full_df = load_and_prepare_data(dataset_url)
    
    # Échantillon de 50 000 tweets pour le workflow CI
    sample_size = min(50000, len(full_df))
    data_df = full_df.sample(n=sample_size, random_state=42)
    
    output_path = os.path.join('data', 'raw_tweets.csv')
    data_df.to_csv(output_path, index=False)
    print(f"Échantillon de données sauvegardé dans {output_path}")
