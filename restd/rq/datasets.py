import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        self.embeddings = np.load(data_path, mmap_mode="r", allow_pickle=False)

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embedding matrix must be rank 2, got {self.embeddings.shape}"
            )
        if self.embeddings.dtype != np.float32:
            raise ValueError(
                f"embedding matrix must be float32, got {self.embeddings.dtype}"
            )
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = np.asarray(self.embeddings[index], dtype=np.float32)
        tensor_emb = torch.tensor(emb, dtype=torch.float32)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)
