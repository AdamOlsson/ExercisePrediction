# custom
from Datasets.GeneralDataset import GeneralDataset
from Transformers.ToTensor import ToTensor
from torch.utils.data import DataLoader

from models.st_gcn.st_gcn_aaai18 import ST_GCN_18

# other
import numpy as np
import torch, torchvision
from torchvision.transforms import Compose


def main(annotations_path):

    transform = [ToTensor(dtype=torch.float32, requires_grad=False)]
    dataset = GeneralDataset(annotations_path, np.load, transform=Compose(transform))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    graph_cfg = {"layout":"openpose", "strategy":"spatial"}
    model = ST_GCN_18(3, len(dataset.labels), graph_cfg, edge_importance_weighting=True, data_bn=True)

    for i_batch, sample_batched in enumerate(dataloader):
        video = sample_batched["data"]
        out = model(video)
        
        print(out)
        exit()

if __name__ == "__main__":
    annotations_path = "../datasets/weightlifting/ndarrays/annotations.csv" 
    main(annotations_path)