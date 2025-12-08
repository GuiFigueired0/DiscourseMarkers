import torch
from torch.utils.data import DataLoader
from models import DynamicMultiTaskModel
from trainer import MultiTaskTrainer
from data_processor import DataProcessor, collate_fn

# Config
MODEL_PATH = "./saved_models/best_multitask_model.pt"
CONFIG = {'model_name': 'roberta-base', 'lr': 2e-5, 'weight_decay': 0.01, 'batch_size': 32, 'max_length': 256}
TASK_CONFIG = {
    'primary': 'anli',
    'secondary': 'dm',
    # Weights don't matter for evaluation, but the init requires the key
    'weights': {'anli': 1.0, 'dm': 0.5}
}

def run_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {MODEL_PATH}...")

    # 1. Re-initialize the Architecture
    model = DynamicMultiTaskModel(CONFIG)
    model.add_task_head('anli', 3)
    model.add_task_head('dm', 4)

    # 2. Load Weights
    # map_location ensures it loads on CPU if GPU is missing
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model loaded successfully.")

    # 3. Setup Data
    processor = DataProcessor(CONFIG)
    # Example: Evaluate on ANLI Round 3
    ds_test = processor.get_anli_dataset(split='test_r1')
    loader = DataLoader(ds_test, batch_size=32, collate_fn=collate_fn)

    # 4. Evaluate
    # We pass None for optimizer/scheduler since we are just evaluating
    trainer = MultiTaskTrainer(model, CONFIG, device, TASK_CONFIG)

    acc, f1 = trainer.evaluate(loader, task_keys='anli')
    print(f"Result on ANLI R3 -> Acc: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    run_eval()