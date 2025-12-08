import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset_a, dataset_b, batch_size):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.batch_size = batch_size
        # Ensure batch size is even
        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling"
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        self.num_batches = min(self.len_a, self.len_b) // (batch_size // 2)

    def __iter__(self):
        # Shuffle indices
        indices_a = torch.randperm(self.len_a).cpu().tolist()
        indices_b = torch.randperm(self.len_b).cpu().tolist()

        # Provide offset for dataset B indices if they are concatenated in a single wrapper dataset
        # But here we will handle data loading distinctly in the Trainer loop

        ptr_a = 0
        ptr_b = 0

        for _ in range(self.num_batches):
            batch = []
            # First half: Task A
            batch.extend(indices_a[ptr_a: ptr_a + self.batch_size // 2])
            ptr_a += self.batch_size // 2

            # Second half: Task B
            batch.extend(indices_b[ptr_b: ptr_b + self.batch_size // 2])
            ptr_b += self.batch_size // 2

            yield batch

    def __len__(self):
        return self.num_batches