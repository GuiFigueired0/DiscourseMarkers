import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_CONFIGS = {
    'anli': {'path': 'facebook/anli', 'split_map': {'train': 'train_r3', 'test': 'test_r3'}, 'cols': ('premise', 'hypothesis', 'label'), 'type': 'pair'},
    'rte': {'path': 'glue', 'name': 'rte', 'cols': ('sentence1', 'sentence2', 'label'), 'type': 'pair'},
    'ag_news': {'path': 'ag_news', 'cols': ('text', None, 'label'), 'type': 'single'},
    'tweet_eval': {'path': 'tweet_eval', 'name': 'hate', 'cols': ('text', None, 'label'), 'type': 'single'},
    'paraphrase': {'path': 'PhilipMay/stsb_multi_mt', 'cols': ('sentence1', 'sentence2', 'similarity_score'), 'type': 'pair'}
}

ARC_LABEL_MAPS = {
    'student_essay': {'supports': 0, 'attacks': 1},
    'debate': {'support': 0, 'attack': 1},
    'm-arg': {'support': 0, 'attack': 1, 'neither': 2}
}

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

class DataProcessor:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.max_len = int(config.get('max_length', 512))
        self.label_encoders = {}  # Store encoders per task to ensure consistency

    def get_dm_partition(self, file_path, test_size=0.1):
        df = pd.read_csv(file_path)

        dm_map = {'CDM': 0, 'EDM': 1, 'IDM': 2, 'TDM': 3, 'SEQ': 2}
        if df['label'].dtype == 'O':
            df['label'] = df['label'].map(dm_map)

        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        print('Number of DM classes:', len(df['label'].unique()))
        print('DM classes:', df['label'].unique())

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
        return self._create_dataset_from_df(train_df), self._create_dataset_from_df(test_df)

    def get_general_dataset(self, task_key, split='train'):
        """Loads HuggingFace Datasets with Manual Split & Label Encoding"""
        cfg = DATASET_CONFIGS[task_key]
        col_text_a, col_text_b, col_label = cfg['cols']

        hf_split = cfg.get('split_map', {}).get(split, split)
        print(f"Loading {task_key} [{hf_split}]...")
        raw_ds = load_dataset(cfg['path'], name=cfg.get('name'), split=hf_split)

        texts_a = [str(t) for t in raw_ds[col_text_a]]
        raw_labels = raw_ds[col_label]

        labels = self._encode_labels(task_key, raw_labels, is_train=('train' in split or 'validation' in split))

        if cfg['type'] == 'pair':
            texts_b = [str(t) for t in raw_ds[col_text_b]]
            return self._create_dataset_lists(texts_a, texts_b, labels)
        else:
            return self._create_dataset_lists(texts_a, None, labels)

    def _encode_labels(self, task_key, raw_labels, is_train=True):
        """Helper to handle string-to-int conversion properly"""
        if len(raw_labels) > 0 and isinstance(list(raw_labels)[0], str):
            print(f"  -> Encoding string labels for {task_key}...")
            if task_key not in self.label_encoders:
                le = LabelEncoder()
                labels = le.fit_transform(raw_labels)
                self.label_encoders[task_key] = le
                print(f"  -> Mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                return labels
            else:
                le = self.label_encoders[task_key]
                if is_train:
                    return le.fit_transform(raw_labels)
                return le.transform(raw_labels)
        return raw_labels

    def _create_dataset_lists(self, texts_a, texts_b, labels):
        if texts_b:
            encodings = self.tokenizer(texts_a, texts_b, truncation=True, padding=True, max_length=self.max_len)
        else:
            encodings = self.tokenizer(texts_a, truncation=True, padding=True, max_length=self.max_len)
        return TextDataset(encodings, labels)

    def _create_dataset_from_df(self, df):
        pairs = df[['s1', 's2']].values.tolist()
        labels = df['label'].tolist()
        encodings = self.tokenizer(pairs, truncation=True, padding=True, max_length=self.max_len)
        return TextDataset(encodings, labels)

    def _split_and_tokenize(self, df, text_col, label_col, test_size):
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df[label_col]
        )

        train_texts = train_df[text_col].astype(str).tolist()
        train_labels = train_df[label_col].astype(int).tolist()
        ds_train = self._create_dataset_lists(train_texts, None, train_labels)

        test_texts = test_df[text_col].astype(str).tolist()
        test_labels = test_df[label_col].astype(int).tolist()
        ds_test = self._create_dataset_lists(test_texts, None, test_labels)

        return ds_train, ds_test

    def get_arc_dataset(self, dataset_name, path_train, path_test):
        def load_file(path):
            s1, s2, labels = [], [], []
            label_map = ARC_LABEL_MAPS[dataset_name]
            if dataset_name == 'm-arg':
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    if row[3].strip() in label_map:
                        s1.append(row[1].strip())
                        s2.append(row[2].strip())
                        labels.append(label_map[row[3].strip()])
            else:
                with open(path, 'r', encoding='ISO-8859-1') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4 and parts[-1].strip() in label_map:
                            s1.append(parts[1].strip())
                            s2.append(parts[3].strip())
                            labels.append(label_map[parts[-1].strip()])
            return self._create_dataset_lists(s1, s2, labels)

        ds_train = load_file(path_train)
        ds_test = load_file(path_test)
        return ds_train, ds_test

    def get_haspeede_dataset(self, file_path, test_size=0.1):
        print(f"Loading HaSpeeDe from {file_path}...")
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")

        df.columns = df.columns.str.strip().str.lower()
        if 'text' not in df.columns or 'hs' not in df.columns:
            raise KeyError(f"Columns 'text' or 'hs' missing in HaSpeeDe.")

        return self._split_and_tokenize(df, 'text', 'hs', test_size)

    def get_folhasp_dataset(self, file_path, test_size=0.1):
        print(f"Loading FolhaSP from {file_path}...")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")

        if 'text' not in df.columns or 'category' not in df.columns:
            raise KeyError(f"FolhaSP columns missing.")

        class_map = {label: idx for idx, label in enumerate(df['category'].unique())}
        df['label'] = df['category'].map(class_map)
        print(f"  -> FolhaSP Classes: {class_map}")

        return self._split_and_tokenize(df, 'text', 'label', test_size)

    def get_hatebr(self, file_path, test_size=0.1):
        print(f"Loading HateBR from {file_path}...")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")

        if 'comentario' not in df.columns:
            raise KeyError(f"HateBR 'comentario' column missing.")

        df['label'] = df['label_final']

        return self._split_and_tokenize(df, 'comentario', 'label', test_size)

    def get_italic(self):
        print(f"Loading ITALIC...")

        target_classes = sorted(
            ['iot', 'calendar', 'play', 'general', 'news', 'weather', 'qa', 'transport', 'email', 'lists'])
        class_map = {label: idx for idx, label in enumerate(target_classes)}

        def load_split(split_name):
            df = pd.read_csv(f'data/italic/massive_{split_name}_filtered.csv')
            df['label'] = df['scenario'].map(class_map)
            df = df.dropna(subset=['label'])
            texts = df['utt'].astype(str).tolist()
            labels = df['label'].astype(int).tolist()
            return self._create_dataset_lists(texts, None, labels)

        ds_train = load_split('train')
        ds_test = load_split('test')
        print(f"  -> ITALIC Consistent Map: {class_map}")

        return ds_train, ds_test

    def get_multilingual_nli_dataset(self, lang_code, test_size=0.1):
        from datasets import concatenate_datasets
        print(f"Loading Multilingual NLI for language '{lang_code}'...")

        subsets = ['anli', 'fever', 'ling', 'mnli', 'wanli']
        dataset_list = []

        for sub in subsets:
            split_name = f"{lang_code if lang_code != 'en' else 'pt'}_{sub}"
            try:
                print(f"  - Loading subset: {split_name}...")
                ds = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", split=split_name)
                dataset_list.append(ds)
            except Exception as e:
                print(f"    ! Warning: Could not load {split_name}: {e}")

        if not dataset_list:
            raise ValueError(f"No datasets found for language {lang_code}.")

        full_ds = concatenate_datasets(dataset_list)
        print(f"  -> Total merged examples: {len(full_ds)}")

        premises = [str(t) for t in full_ds['premise' if lang_code != 'en' else 'premise_original']]
        hypotheses = [str(t) for t in full_ds['hypothesis' if lang_code != 'en' else 'hypothesis_original']]
        labels = full_ds['label']

        train_p, test_p, train_h, test_h, train_y, test_y = train_test_split(
            premises, hypotheses, labels,
            test_size=test_size,
            random_state=42,
            stratify=labels
        )

        return self._create_dataset_lists(train_p, train_h, train_y), self._create_dataset_lists(test_p, test_h, test_y)

    def get_paraphrase_dataset(self, lang_code, threshold=3.0):
        print(f"Loading Paraphrase Dataset (STSb) for language '{lang_code}'...")

        def load_split(split_name):
            raw_ds = load_dataset("PhilipMay/stsb_multi_mt", name=lang_code, split=split_name)
            s1 = [str(t) for t in raw_ds['sentence1']]
            s2 = [str(t) for t in raw_ds['sentence2']]
            scores = raw_ds['similarity_score']
            labels = [1 if s >= threshold else 0 for s in scores]
            return self._create_dataset_lists(s1, s2, labels)

        # STSb has fixed splits: train and test
        ds_train = load_split('train')
        ds_test = load_split('test')

        return ds_train, ds_test

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=1)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}