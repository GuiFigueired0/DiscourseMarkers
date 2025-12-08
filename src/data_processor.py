import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Map for Discourse Markers (Same as your reference)
DM_CLASS_MAP = {'CDM': 0, 'EDM': 1, 'IDM': 2, 'TDM': 3}


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class DataProcessor:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.max_len = int(config.get('max_length', 256))

        print("Pad token value:", self.tokenizer.pad_token_id)

    def get_anli_dataset(self, split='train'):
        """Loads ANLI from HuggingFace"""
        # Load specific split (e.g., 'train_r1', 'test_r1')
        dataset = load_dataset("facebook/anli", split=split)

        texts_pair = [[e['premise'], e['hypothesis']] for e in dataset]
        labels = [e['label'] for e in dataset]

        encodings = self.tokenizer(texts_pair, truncation=True, padding=True, max_length=self.max_len)
        return TextDataset(encodings, labels)

    def get_dm_partition(self, file_path, test_size=0.1):
        """
        Loads DM CSV, maps labels to IDs, and splits into Train and Test.
        Returns: (train_dataset, test_dataset)
        """
        df = pd.read_csv(file_path)

        # Ensure labels are integers. If strings (CDM, etc.), map them.
        if df['label'].dtype == 'O':
            df['label'] = df['label'].map(DM_CLASS_MAP)

        # Drop any rows that failed mapping
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        # Split Dataframe
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])

        def create_dataset(dataframe):
            texts_pair = dataframe[['s1', 's2']].values.tolist()
            labels = dataframe['label'].tolist()
            encodings = self.tokenizer(texts_pair, truncation=True, padding=True, max_length=self.max_len)
            return TextDataset(encodings, labels)

        return create_dataset(train_df), create_dataset(test_df)


def collate_fn(batch):
    # 1. Extract input_ids and mask from the list of items
    # We assume item['input_ids'] is a 1D tensor like [101, 54, ... 102]
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    # 3. Stack the labels (labels are usually single integers, so stack works fine here)
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }