from torch.utils.data import Dataset

# Create graph-dataset
# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html

# create_graph as an example
# https://github.com/dawidejdeholm/dj_graph/blob/master/Utils/SECParser.py

# install
# https://networkx.github.io/

class GraphDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        def construct_graph():
            pass
        pass

