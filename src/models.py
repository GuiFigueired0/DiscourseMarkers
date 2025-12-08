import torch.nn as nn
from transformers import AutoModel

class DynamicMultiTaskModel(nn.Module):
    def __init__(self, config):
        super(DynamicMultiTaskModel, self).__init__()

        # 1. The Shared Encoder (The "Discourse Marker" Base)
        self.encoder = AutoModel.from_pretrained(config['model_name'])
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # 2. Dynamic Dictionary for Task Heads
        # ModuleDict allows us to add sub-modules by string keys
        self.heads = nn.ModuleDict()

    def add_task_head(self, task_name, num_labels):
        """
        Call this method from main.py to register a new task.
        """
        print(f"-> Adding Task Head: '{task_name}' with {num_labels} labels.")
        self.heads[task_name] = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, task_keys=None):
        """
        task_keys: A list or string indicating which head to use.
                   If it's a list (e.g., ['anli', 'dm']), we assume the batch
                   is split: top half = 'anli', bottom half = 'dm'.
        """
        # 1. Shared Encoder Pass
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # 2. Dynamic Routing
        if isinstance(task_keys, str):
            # Case A: Pure batch (Evaluation) -> Single task
            if task_keys not in self.heads:
                raise ValueError(f"Task '{task_keys}' not registered.")
            return self.heads[task_keys](cls_embedding)

        elif isinstance(task_keys, list):
            # Case B: Mixed batch (Training) -> Split batch logic
            # Assumes batch is perfectly balanced 50/50 between the two tasks
            batch_size = cls_embedding.size(0)
            split_idx = batch_size // 2

            task_a_name = task_keys[0]
            task_b_name = task_keys[1]

            emb_a = cls_embedding[:split_idx]
            emb_b = cls_embedding[split_idx:]

            out_a = self.heads[task_a_name](emb_a)
            out_b = self.heads[task_b_name](emb_b)

            return out_a, out_b

        else:
            return cls_embedding  # Return raw embeddings if no task specified