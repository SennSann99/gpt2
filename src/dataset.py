import torch
from torch.utils.data import Dataset

class TextChunkDataset(Dataset):
    """
    Creates overlapping token chunks for autoregressive language modeling.
    """
    def __init__(self, texts_series, tokenizer, max_len, stride):
        self.chunks = []
        # TODO: For production with large datasets, replace this in-memory list 
        # with an IterableDataset or HuggingFace memory-mapped datasets to prevent OOM.
        for text in texts_series:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - max_len, stride):
                self.chunks.append(torch.tensor(tokens[i : i + max_len + 1]))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # Returns (input_ids, target_ids)
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]