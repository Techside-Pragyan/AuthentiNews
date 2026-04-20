import pandas as pd
import requests
import os

def download_dataset():
    data_url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
    data_path = "data/datasets/fake_or_real_news.csv"
    
    if not os.path.exists(data_path):
        print(f"Downloading dataset from {data_url}...")
        response = requests.get(data_url)
        with open(data_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def prepare_data():
    data_path = "data/datasets/fake_or_real_news.csv"
    df = pd.read_csv(data_path)
    
    # Simple cleaning
    df = df.dropna()
    
    # Map label to numeric
    # Assuming labels are 'REAL' and 'FAKE'
    df['label_num'] = df['label'].apply(lambda x: 1 if x == 'REAL' else 0)
    
    # Save a small subset for quick training/testing if needed
    train_path = "data/datasets/train_processed.csv"
    df.to_csv(train_path, index=False)
    print(f"Prepared data saved to {train_path}")

if __name__ == "__main__":
    if not os.path.exists("data/datasets"):
        os.makedirs("data/datasets")
    download_dataset()
    prepare_data()
