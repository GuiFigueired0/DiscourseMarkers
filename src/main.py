import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from sklearn.utils.class_weight import compute_class_weight

from models import DynamicMultiTaskModel
from trainer import MultiTaskTrainer
from batch_sampler import BalancedBatchSampler
from data_processor import DataProcessor, collate_fn

# --- CONFIGURATION ---
CONFIG = {
    'model_name': 'roberta-base',
    'lr': 2e-5,
    'batch_size': 32,
    'epochs': 10,
    'dm_csv_path': './../data/dm_en.csv',
    'model_dir': './saved_models',
    'early_stopping': 2
}

def get_class_weights(dataset):
    # Extract labels from the dataset (assuming dataset.labels is a list/tensor)
    labels = dataset.labels
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = DataProcessor(CONFIG)

    # 1. Initialize Model
    print(f"--- Mode: {args.mode} | Step: {args.step} ---")
    model = DynamicMultiTaskModel(CONFIG)
    model.add_task_head('anli', 3)
    model.add_task_head('dm', 4)
    model.to(device)

    # 2. Logic: Load Pre-trained Weights if not starting from scratch
    prev_model_path = None
    if args.step > 0:
        prev_step_name = f"{args.mode}_step{args.step - 1}.pt"
        prev_model_path = os.path.join(CONFIG['model_dir'], prev_step_name)

    if prev_model_path and os.path.exists(prev_model_path):
        print(f"Loading checkpoint: {prev_model_path}")
        model.load_state_dict(torch.load(prev_model_path, map_location=device))
    elif args.step > 0:
        print(f"Warning: Previous checkpoint {prev_model_path} not found! Starting fresh.")

    # 3. Prepare Data & Trainer based on Mode
    task_weights = None
    test_ds = None
    is_dm_pre_training = False

    # --- MODE: MTL (Multi-Task Learning) ---
    if args.mode == 'mtl':
        ds_anli = processor.get_anli_dataset(split=f'train_r{args.step}')
        ds_dm, _ = processor.get_dm_partition(CONFIG['dm_csv_path'])

        # Compute DM Weights
        dm_weights = get_class_weights(ds_dm)
        task_weights = {'dm': dm_weights}

        # Sampler
        sampler = BalancedBatchSampler(ds_anli, ds_dm, CONFIG['batch_size'])
        combined = ConcatDataset([ds_anli, ds_dm])
        train_loader = DataLoader(combined, batch_sampler=sampler, collate_fn=collate_fn)
        task_mode = 'mixed'

    # --- MODE: BASELINE or TRANSFER (Single Task Steps) ---
    else:
        # Step 0: Pre-training on DM (Only for Transfer mode)
        if args.mode == 'transfer' and args.step == 0:
            print("Setup: Training on Discourse Markers (DM)")
            ds_train, test_ds = processor.get_dm_partition(CONFIG['dm_csv_path'])
            dm_weights = get_class_weights(ds_train)
            task_weights = {'dm': dm_weights}
            task_mode = 'dm'
            is_dm_pre_training = True

        # Step 1, 2, 3: ANLI Rounds
        else:
            round_key = f"train_r{args.step}"  # e.g., train_r1
            print(f"Setup: Training on ANLI {round_key}")
            ds_train = processor.get_anli_dataset(split=round_key)
            task_mode = 'anli'

        train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)

    if test_ds is None:
        test_ds = processor.get_anli_dataset(split=f'test_r{args.step}')
    if args.mode != 'baseline':
        _, dm_test = processor.get_dm_partition(CONFIG['dm_csv_path'])

    # 4. Initialize Trainer
    trainer_config = {
        'primary': 'anli', 'secondary': 'dm',
        'weights': {'anli': 1.0, 'dm': 0.5}  # Only used if task_mode='mixed'
    }
    trainer = MultiTaskTrainer(model, CONFIG, device, trainer_config, class_weights=task_weights)

    # 5. Training Loop
    best_avg_acc = 0.0
    save_name = f"{args.mode}_step{args.step}.pt"
    best_epoch = -1

    print(f"Starting Training ({task_mode})...")
    for epoch in range(CONFIG['epochs']):
        if epoch - best_epoch > CONFIG['early_stopping']:
            print(f"Early stopping at epoch {best_epoch}")
            break

        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        loss = trainer.train_epoch(train_loader, epoch + 1, task_mode=task_mode)
        print(f"  > Average Train Loss: {loss:.4f}")

        # --- EVALUATION ---
        print("  > Evaluating...")

        acc, f1 = trainer.evaluate(
            DataLoader(test_ds, batch_size=32, collate_fn=collate_fn),
            'dm' if is_dm_pre_training else 'anli'
        )
        print(f"  >> Acc: {acc:.4f} | F1: {f1:.4f}")

        if args.mode != 'baseline' and args.step:
            dm_acc, dm_f1 = trainer.evaluate(DataLoader(dm_test, batch_size=32, collate_fn=collate_fn), 'dm')
            print(f"  >> [DM Test] Acc: {acc:.4f} | F1: {f1:.4f}")

        # --- CHECKPOINTING ---
        if acc > best_avg_acc:
            best_epoch = epoch
            best_avg_acc = acc
            save_path = os.path.join(CONFIG['model_dir'], save_name)
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Model Saved! (Avg Acc: {best_avg_acc:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['mtl', 'baseline', 'transfer'], required=True,
                        help="Experiment mode")
    parser.add_argument('--step', type=int, default=1,
                        help="Step number (1=R1, 2=R2, 3=R3). For transfer, 0=DM pretrain.")
    args = parser.parse_args()

    run_experiment(args)