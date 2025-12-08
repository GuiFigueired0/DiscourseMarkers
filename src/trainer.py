import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class MultiTaskTrainer:
    def __init__(self, model, config, device, task_configs, class_weights=None):
        """
        class_weights: Dict mapping task_name to tensor of weights.
                       e.g. {'dm': torch.tensor([1.0, 5.0, ...])}
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.task_configs = task_configs

        self.optimizer = AdamW(model.parameters(), lr=float(config['lr']), weight_decay=0.01)

        # Store class weights if provided
        self.loss_fcts = {}
        for task, weights in (class_weights or {}).items():
            if weights is not None:
                self.loss_fcts[task] = nn.CrossEntropyLoss(weight=weights.to(device))

        # Default loss for tasks without specific weights
        self.default_loss_fct = nn.CrossEntropyLoss()

    def get_loss_fct(self, task_name):
        return self.loss_fcts.get(task_name, self.default_loss_fct)

    def train_epoch(self, dataloader, epoch_idx, task_mode='mixed'):
        """
        task_mode: 'mixed' (MTL), 'anli' (Single), 'dm' (Single)
        """
        self.model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch_idx} [{task_mode}]")

        for batch in loop:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # --- CASE 1: Single Task Training ---
            if task_mode != 'mixed':
                # Forward full batch to one head
                logits = self.model(input_ids, attention_mask, task_keys=task_mode)
                loss_fct = self.get_loss_fct(task_mode)
                loss = loss_fct(logits, labels)

            # --- CASE 2: Multi-Task Learning (Mixed Batch) ---
            else:
                task_a = self.task_configs['primary']
                task_b = self.task_configs['secondary']

                # Forward split
                logits_a, logits_b = self.model(input_ids, attention_mask, task_keys=[task_a, task_b])

                # Label split
                half = labels.size(0) // 2
                loss_a = self.get_loss_fct(task_a)(logits_a, labels[:half])
                loss_b = self.get_loss_fct(task_b)(logits_b, labels[half:])

                loss = (self.task_configs['weights'][task_a] * loss_a) + \
                       (self.task_configs['weights'][task_b] * loss_b)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(dataloader)

    def evaluate(self, dataloader, task_keys):
        self.model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval {task_keys}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask, task_keys=task_keys)
                preds = torch.argmax(logits, dim=1)
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        return accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average='macro')