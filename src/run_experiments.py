import argparse
import os
import torch
import csv
import gc
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from models import DynamicMultiTaskModel
from trainer import MultiTaskTrainer
from batch_sampler import BalancedBatchSampler
from data_processor import DataProcessor, collate_fn
from sklearn.utils.class_weight import compute_class_weight

# --- MASTER CONFIGURATION ---
CONFIG = {
    'max_length': 512,
    'batch_size': 64,
    'lr': 1e-5,
    'weight_decay': 0.01,
    'epochs': 30,
    'early_stopping': 5,
    'model_dir': './saved_models',
    'results_file': 'results.csv'
}

TASK_META = {
    # --- Multilingual & Custom ---
    'nli_multi': {'n_labels': 3, 'type': 'nli_custom'},
    'paraphrase': {'n_labels': 2, 'type': 'paraphrase_custom'},

    # --- HuggingFace Standard ---
    'ag_news': {'n_labels': 4, 'type': 'hf'},
    'tweet_eval': {'n_labels': 2, 'type': 'hf'},
    'anli': {'n_labels': 3, 'type': 'hf'},
    'rte': {'n_labels': 2, 'type': 'hf'},

    # --- Local Standard (Load File -> Split) ---
    'haspeede': {
        'n_labels': 2, 'type': 'local_standard',
        'path': './data/haspeede2_dev/haspeede2_dev_taskAB.tsv',
        'method': 'get_haspeede_dataset'
    },
    'folhasp': {
        'n_labels': 18, 'type': 'local_standard',
        'path': './data/FolhaSP/articles_filtered.csv',
        'method': 'get_folhasp_dataset'
    },
    'hatebr': {
        'n_labels': 2, 'type': 'local_standard',
        'path': './data/HateBR/HateBR.csv',
        'method': 'get_hatebr'
    },

    # --- Local Specialized (Custom Logic) ---
    'italic': {'n_labels': 10, 'type': 'local_italic'},
    'debate': {'n_labels': 2, 'type': 'local_arc'},
}

TASK_TYPE = {
    'Topic Classification': 'Non-NLI',
    'Semantic Textual Similarity': 'NLI-Like',
    'Natural Language Inference': 'NLI-Like',
    'Hate Speech Detection': 'Non-NLI'
}

DATASET_TO_TASK = {
    'nli_multi': 'Natural Language Inference',
    'paraphrase': 'Semantic Textual Similarity',
    'ag_news': 'Topic Classification',
    'tweet_eval': 'Hate Speech Detection',
    'haspeede': 'Hate Speech Detection',
    'italic': 'Topic Classification',
    'folhasp': 'Topic Classification',
    'hatebr': 'Hate Speech Detection',
    'anli': 'Natural Language Inference',
    'debate': 'Natural Language Inference',
    'rte': 'Natural Language Inference',
}

LANGUAGE_MAP = {
    'en': {
        'num_dm_classes': 4,
        'model_name': 'roberta-base',
    },
    'it': {
        'num_dm_classes': 3,
        'model_name': 'Musixmatch/umberto-commoncrawl-cased-v1'
    },
    'pt': {
        'num_dm_classes': 3,
        'model_name': 'neuralmind/bert-base-portuguese-cased'
    },
}

def log_result(data):
    file_exists = os.path.isfile(CONFIG['results_file'])
    with open(CONFIG['results_file'], mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def get_class_weights(dataset):
    labels = dataset.labels
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)

def run_task(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_name = args.task
    meta = TASK_META[task_name]
    model_name = LANGUAGE_MAP[args.language]['model_name']
    CONFIG['model_name'] = model_name

    processor = DataProcessor(CONFIG)

    print(f"\n" + "=" * 75)
    print(f"STARTING EXPERIMENT: Task={task_name} | Mode={args.mode} | Language={args.language}")
    print("=" * 75)

    if meta['type'] == 'local_standard':
        loader = getattr(processor, meta['method'])
        ds_train, ds_test = loader(meta['path'], test_size=0.1)

    elif meta['type'] == 'nli_custom':
        ds_train, ds_test = processor.get_multilingual_nli_dataset(args.language, test_size=0.1)

    elif meta['type'] == 'paraphrase_custom':
        ds_train, ds_test = processor.get_paraphrase_dataset(args.language)

    elif meta['type'] == 'local_italic':
        ds_train, ds_test = processor.get_italic()

    elif meta['type'] == 'local_arc':
        prefix = 'essay' if task_name == 'student_essay' else 'debate_concept'
        path_train = f"./../data/{task_name}/train_{prefix}.txt"
        path_test = f"./../data/{task_name}/test_{prefix}.txt"
        ds_train, ds_test = processor.get_arc_dataset(task_name, path_train, path_test)

    else:
        # HuggingFace logic
        ds_train = processor.get_general_dataset(task_name, split='train')
        test_split = 'validation' if task_name == 'rte' else 'test'
        ds_test = processor.get_general_dataset(task_name, split=test_split)

    ds_dm_train, _ = processor.get_dm_partition(f'./../data/dm_{args.language}.csv')

    betas = [0.2, 0.4, 0.6, 0.8, 1.0] if args.mode == 'mtl' else [0.0]

    class_weights_dict = {
        task_name: get_class_weights(ds_train),
        'dm': get_class_weights(ds_dm_train)
    }

    best_beta = -1
    best_global_f1 = 0.0
    best_global_acc = 0.0

    for beta in betas:
        print(f"\n--- Running with Beta: {beta} ---")

        model = DynamicMultiTaskModel(CONFIG)

        if args.mode == 'transfer':
            print(">> [Transfer] Loading pre-trained DM weights...")
            model.add_task_head('dm', LANGUAGE_MAP[args.language]['num_dm_classes'])
            dm_path = os.path.join(CONFIG['model_dir'], f'best_dm_model_{args.language}.pt')
            if os.path.exists(dm_path):
                model.load_state_dict(torch.load(dm_path, map_location=device))
            else:
                print("!! Warning: DM Model not found. Training from scratch.")

        model.add_task_head(task_name, meta['n_labels'])
        if 'dm' not in model.heads:
            model.add_task_head('dm', LANGUAGE_MAP[args.language]['num_dm_classes'])

        model.to(device)
        task_mode = task_name

        if args.mode == 'mtl' and beta > 0:
            sampler = BalancedBatchSampler(ds_train, ds_dm_train, CONFIG['batch_size'])
            combined = ConcatDataset([ds_train, ds_dm_train])
            train_loader = DataLoader(combined, batch_sampler=sampler, collate_fn=collate_fn)
            task_mode = 'mixed'
        else:
            train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)

        trainer_config = {
            'primary': task_name,
            'secondary': 'dm',
            'weights': {task_name: 1.0, 'dm': beta}
        }

        trainer = MultiTaskTrainer(model, CONFIG, device, trainer_config, class_weights=class_weights_dict)

        best_f1 = 0.0
        best_epoch = -1
        for epoch in range(CONFIG['epochs']):
            loss = trainer.train_epoch(train_loader, epoch + 1, task_mode=task_mode)

            test_acc, test_f1 = trainer.evaluate(DataLoader(ds_test, batch_size=32, collate_fn=collate_fn), task_name)
            print(f"  Ep {epoch + 1} | Loss: {loss:.4f} | Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

            if test_f1 > best_global_f1:
                best_global_f1 = test_f1
                best_global_acc = test_acc
                best_beta = beta

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch

            if epoch - best_epoch >= CONFIG['early_stopping']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # --- MEMORY CLEANUP ---
        del model
        del trainer
        del train_loader
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory cleared for Beta {beta}")
        # ----------------------

    log_result({
        'language': args.language,
        'task': DATASET_TO_TASK[task_name],
        'type': TASK_TYPE[DATASET_TO_TASK[task_name]],
        'dataset': task_name,
        'mode': args.mode,
        'beta': best_beta,
        'final_test_acc': best_global_acc,
        'final_test_f1': best_global_f1
    })
    print(f"Done. Logged to {CONFIG['results_file']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=[
                            'nli_multi', 'paraphrase'   # Multilingual
                                         
                            'ag_news', 'tweet_eval',    # English
                            'haspeede', 'italic',       # Italian
                            'folhasp', 'hatebr',        # Portuguese

                            'rte', 'anli', 'debate',    # Extras
                        ])
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'transfer', 'mtl'])
    parser.add_argument('--language', type=str, required=True, choices=['en', 'it', 'pt'])
    args = parser.parse_args()

    run_task(args)
