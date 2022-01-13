import numpy as np
from torch.utils.data import Dataset
import pickle


class LogitsLabelsDataset(Dataset):
    """logits and labels dataset."""

    def __init__(self, root_dir, type='val'):
        """
        Args:
            root_dir (string): Directory with all the logits and labels.
        """
        with open(root_dir + '/' + type + '_labels.pickle', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(root_dir + '/' + type + '_logits.pickle', 'rb') as handle:
            self.logits = pickle.load(handle)
            
    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        
        return self.logits[idx], self.labels[idx].long()