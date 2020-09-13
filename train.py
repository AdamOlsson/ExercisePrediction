# custom
from Datasets.GeneralDataset import GeneralDataset
from Transformers.ToTensor import ToTensor
from torch.utils.data import DataLoader

# model and loss
from models.st_gcn.st_gcn_aaai18 import ST_GCN_18
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

# other
import numpy as np
import torch, torchvision
from torchvision.transforms import Compose

def oneHotEncodeLabels(labels):
    """Takes a list of string labels and one-hot encodes each item.
    
    Parameters:
        labels - (list) list of string labels

    Return:
        (dict) a dict with each label as a key and respective one-hot encoding
    """
    one_hot = {}
    for i, label in enumerate(labels):
        one_hot[label] = i

    return one_hot

def batchLabels(one_hot, labels):
    return torch.tensor([one_hot[labels[0]]], requires_grad=False, dtype=torch.long)

def main(annotations_path):

    # Hyperparameters
    device   = "cuda"        # device
    lr = 0.01
    momentum = 0.9
    decay = 0.0001
    loss_fn  = CrossEntropyLoss()  # loss
    layout   = "openpose"
    strategy = "spatial"

    transform = [ToTensor(dtype=torch.float32, requires_grad=False, device=device)]
    dataset = GeneralDataset(annotations_path, np.load, transform=Compose(transform))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # one-hot encode labels for loss computations
    labels = oneHotEncodeLabels(dataset.labels)
    
    graph_cfg = {"layout":layout, "strategy":strategy}
    model = ST_GCN_18(3, len(dataset.labels), graph_cfg, edge_importance_weighting=True, data_bn=True).to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay, nesterov=True)
    model.train()

    # TODO: Epochs
    # TODO: Train/Test split
    losses = []
    for i_batch, sample_batched in enumerate(dataloader):
        video = sample_batched["data"]
        label = batchLabels(labels, sample_batched["label"]).to(device)
        # TODO Fix batch labeling
        # TODO: adjust lr
        output = model(video)
        loss = loss_fn(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data.item())

        if i_batch % 5 == 0:
            print("Step {}, Mean loss: {}".format((i_batch), np.mean(losses)))

    print("Mean loss after training: {}".format(np.mean(losses)))
        


if __name__ == "__main__":
    annotations_path = "../datasets/weightlifting/ndarrays/annotations.csv" 
    main(annotations_path)