import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from models import DynamicMultiTaskModel
from trainer import MultiTaskTrainer
from data_processor import DataProcessor, collate_fn

CONFIG = {
    'max_length': 512,
    'batch_size': 64,
    'lr': 1e-5,
    'weight_decay': 0.01,
    'epochs': 30,
    'early_stopping': 5,
    'model_dir': './saved_models'
}

MODEL_NAME = {
    'en': 'roberta-base',
    'it': 'Musixmatch/umberto-commoncrawl-cased-v1',
    'pt': 'neuralmind/bert-base-portuguese-cased',
}

def get_class_weights(dataset):
    # Calculate balanced weights for DM classes
    labels = dataset.labels
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float), len(classes)

def run_dm_training(args):
    CONFIG['model_name'] = MODEL_NAME[args.language]
    print(f"Model name: {CONFIG['model_name']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(CONFIG['model_dir']):
        os.makedirs(CONFIG['model_dir'])

    print(f"--- Starting Discourse Marker (DM) Pre-training ---")
    processor = DataProcessor(CONFIG)
    ds_train, ds_test = processor.get_dm_partition(f'./../data/dm_{args.language}.csv')

    dm_weights, num_classes = get_class_weights(ds_train)
    print(f"DM Class Weights: {dm_weights}. Number of classes: {num_classes}")

    train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(ds_test, batch_size=CONFIG['batch_size'], collate_fn=collate_fn)

    model = DynamicMultiTaskModel(CONFIG)
    model.add_task_head('dm', num_classes)
    model.to(device)

    trainer_config = {'primary': 'dm', 'secondary': None, 'weights': None}
    class_weights_dict = {'dm': dm_weights}

    trainer = MultiTaskTrainer(model, CONFIG, device, trainer_config, class_weights=class_weights_dict)

    best_f1 = 0.0
    best_epoch = -1
    for epoch in range(CONFIG['epochs']):
        loss = trainer.train_epoch(train_loader, epoch + 1, task_mode='dm')

        acc, f1 = trainer.evaluate(test_loader, 'dm')
        print(f"Epoch {epoch + 1} | Loss: {loss:.4f} | Test Acc: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            save_path = os.path.join(CONFIG['model_dir'], f'best_dm_model_{args.language}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best DM Model Saved! (Acc: {acc:.4f} | F1: {best_f1:.4f})")

        if epoch - best_epoch >= CONFIG['early_stopping']:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("DM Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True, choices=['en', 'it', 'pt'])

    args = parser.parse_args()
    run_dm_training(args)