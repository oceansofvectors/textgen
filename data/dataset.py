import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_seq, seq_length):
        self.input_seq = input_seq
        self.seq_length = seq_length

    def __len__(self):
        return len(self.input_seq) - self.seq_length
    
    def __getitem__(self, idx):
        return (torch.tensor(self.input_seq[idx:idx + self.seq_length], dtype=torch.long), 
                torch.tensor(self.input_seq[idx + 1:idx + self.seq_length + 1], dtype=torch.long))
    
