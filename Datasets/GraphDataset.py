from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        pass

