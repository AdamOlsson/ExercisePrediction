import matplotlib.pyplot as plt
import numpy as np

# torch and torchvision
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.io import read_video
from torchvision.transforms import Compose

# Custom Transformers
from transformer.RandomCrop import RandomCrop
from transformer.ToTensor import ToTensor

# Custom Datasets
from Dataset.ImageDataset import ImageDataset
 
def showRandomSample(dataset):
    """
    Sample and display 4 random samples from the dataset.
    """
    fig, ax = plt.subplots(1,4, figsize=(15,8))
 
    for i in range(4):
        rnd = np.random.randint(len(dataset))
        item = dataset[rnd]
        image, label = item['image'], item['label']

        if isinstance(item['image'], torch.Tensor):
            image = image.numpy().transpose((1,2,0))
        
        ax[i].imshow(image)
        ax[i].set_title(item['label'], fontsize=20, fontweight='bold') 
 
    plt.show()


if __name__ == "__main__":
    path_data = "../data/images/"

    path_annotations = "../data/images/annotations.csv"
    dataset = ImageDataset(path_annotations, path_data)
    # dataset = ImageDataset(path_annotations, path_data, transorm=Compose([RandomCrop(224), ToTensor()]))
    
    showRandomSample(dataset)

    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)