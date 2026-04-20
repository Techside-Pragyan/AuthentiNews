import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train():
    data_path = "data/datasets/train_processed.csv"
    if not os.path.exists(data_path):
        print("Dataset not found. Run prepare_data.py first.")
        return

    df = pd.read_csv(data_path)
    # Using a small subset for demonstration if data is too large
    # df = df.sample(1000) 

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label_num'].tolist(), test_size=0.2
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    print("Tokenizing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    print("Starting training (1 epoch for demo)...")
    model.train()
    for epoch in range(1):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}", end='\r')

    # Save model
    model_save_path = "models/trained/authentinews_distilbert"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    # Note: Training might take significant time/resources.
    # For the immediate demo, we might skip this and use a pre-trained model in the API.
    print("This script will fine-tune DistilBERT on the provided news dataset. This requires a GPU for reasonable speed.")
    # train() # Commented out by default to avoid hanging the environment, instructions will suggest running it.
