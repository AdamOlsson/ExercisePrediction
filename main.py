# custom
from Datasets.GeneralDataset import GeneralDataset

# native
import numpy as np

def main(annotations_path):
    dataset = GeneralDataset(annotations_path, np.load)
    
    for i in range(len(dataset)):
        data = dataset[i]
        print(type(data["data"]))


if __name__ == "__main__":
    annotations_path = "../datasets/weightlifting/ndarrays/annotations.csv" 
    main(annotations_path)