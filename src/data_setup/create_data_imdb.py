import logging
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm  # Progress bar
from utils.project import set_root

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMDb Dataset class
class IMDBDataset(Dataset):
    def __init__(self, dataset, text_pipeline):
        self.dataset = dataset
        self.text_pipeline = text_pipeline

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example["text"]
        processed_text = self.text_pipeline(text)
        metadata = {"label": example["label"]}
        return processed_text, metadata, text

# Collate function for batching
def collate_batch(batch):
    input_ids_list, metadata_list, raw_text_list = [], [], []
    for (_input_ids, _metadata, _raw_text) in batch:
        input_ids_list.append(_input_ids)
        metadata_list.append(_metadata)
        raw_text_list.append(_raw_text)
    input_ids_tensor = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True)
    attention_mask = (input_ids_tensor != 0).long()  # Create attention mask
    return input_ids_tensor, attention_mask, metadata_list, raw_text_list

# Load and preprocess the IMDb dataset
def load_imdb_data():
    logger.info("Loading IMDb dataset from the datasets module...")
    dataset = load_dataset('imdb')  # Load dataset

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def text_pipeline(text):
        return tokenizer(
            text, padding='max_length', truncation=True, max_length=128, return_tensors='pt'
        )['input_ids'].squeeze(0)
    
    return dataset, text_pipeline

# Main workflow
def main():
    set_root()
    # Load dataset and tokenizer
    dataset, text_pipeline = load_imdb_data()
    train_dataset = IMDBDataset(dataset["train"], text_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

    # Load pretrained model
    logger.info("Loading pretrained model...")
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    # Extract hidden states and combine with metadata
    logger.info("Extracting hidden states and combining with metadata...")
    data = []
    with tqdm(total=len(train_loader), desc="Processing Batches") as pbar:
        for batch in train_loader:
            input_ids, attention_mask, metadata_list, raw_texts = batch
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_hidden_states = outputs.last_hidden_state[:, 0, :]  # Extract CLS token's hidden state
                for i in range(len(raw_texts)):
                    entry = {
                        "text": raw_texts[i],
                        "hidden_state": cls_hidden_states[i].tolist(),
                        "label": metadata_list[i]["label"]
                    }
                    data.append(entry)
            pbar.update(1)  # Update progress bar

    # Save to CSV
    logger.info("Saving data to CSV...")
    save_dir = "data/imdb"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "imdb_with_hidden_states_sentiment.csv")
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    logger.info(f"Data saved to '{save_path}'.")

if __name__ == "__main__":
    main()
