# torch
from torch.utils.data import Dataset
from pandas import read_csv

# native
from os.path import join, dirname

class GeneralDataset(Dataset):
    '''A general dataset for loading data. To be able to handle different types of data, a
    load function needs to be provide upon initialization.

    The class assumes that the first value in the csv annotations file is the filename.

    Arguements
        annotations_path (String) - path to the annotations file
        load_fn (function) - a function that takes a filename (string) as arguement
                             and loads the data.
    '''
    def __init__(self, annotations_path, load_fn, transform=None):
        self.root = dirname(annotations_path)
        self.annotations = read_csv(annotations_path)
        self.load = load_fn
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        name = join(self.root, self.annotations.iloc[idx, 0])
        label = self.annotations.iloc[idx, 1]
        data = self.load(name)

        if not self.transform == None:
            data = self.transform(data)

        return {"data":data, "label":label}