# custom
from Datasets.GeneralDataset import GeneralDataset
from Transformers.ToTensor import ToTensor
from torch.utils.data import DataLoader, random_split
from PosePrediction.util.load_config import load_config

# model and loss
from models.st_gcn.st_gcn_aaai18 import ST_GCN_18
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
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

    config = load_config("config.json")

    # Hyperparameters
    device      = config["train"]["device"]
    layout      = config["train"]["layout"]
    strategy    = config["train"]["strategy"]
    lr          = config["train"]["lr"]
    gamma       = config["train"]["gamma"]
    momentum    = config["train"]["momentum"]
    decay       = config["train"]["decay"]
    test_split  = config["train"]["test_split"]

    loss_fn  = CrossEntropyLoss()

    exclude_classes = [ "clean", "clean_and_jerk", "clean_pull", "jerk",
                        "other", "power_clean_and_jerk", "power_clean_power_jerk", "power_jerk",
                        "power_snatch", "power_snatch_and_snatch", "push_press_and_jerk", "snatch_and_power_snatch",
                        "snatch_balance", "snatch_pull", "squat_jerk", "clean_power_jerk"] # prototyping purpose

    transform = [ToTensor(dtype=torch.float32, requires_grad=False, device=device)]
    dataset = GeneralDataset(annotations_path, np.load, transform=Compose(transform), classes_to_exclude=exclude_classes)

    test_len  = int(len(dataset)*test_split)
    train_len = len(dataset)-test_len

    trainset, testset = random_split(dataset, [train_len, test_len])

    dataloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

    # one-hot encode labels for loss computations
    labels = oneHotEncodeLabels(dataset.labels)
    
    graph_cfg = {"layout":layout, "strategy":strategy}
    model = ST_GCN_18(3, len(dataset.labels), graph_cfg, edge_importance_weighting=True, data_bn=True).to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay, nesterov=True)
    lr_scheduler = StepLR(optimizer, 20, gamma=gamma)
    model.train()

    # TODO: Epochs
    # TODO: Train/Test split
    losses = []
    for i_batch, sample_batched in enumerate(dataloader):
        video = sample_batched["data"]
        label = batchLabels(labels, sample_batched["label"]).to(device)
        # TODO Fix batch labeling
        output = model(video)
        loss = loss_fn(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        losses.append(loss.data.item())

        if i_batch % 5 == 0:
            print("Step {}, Mean loss: {}".format((i_batch), np.mean(losses)))

    print("Mean loss after training: {}".format(np.mean(losses)))
        


if __name__ == "__main__":
    annotations_path = "../datasets/weightlifting/ndarrays/annotations.csv" 
    main(annotations_path)