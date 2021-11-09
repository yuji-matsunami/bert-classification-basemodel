from torch.utils.data import Dataset
import torch

class ABSADataset(Dataset):
    def __init__(self, encodings, label: list) -> None:
        self.encodings = encodings
        self.label = label
    def __len__(self) -> int:
        return len(self.label)
    
    def __getitem__(self, index:int) -> dict:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.label[index])
        return item