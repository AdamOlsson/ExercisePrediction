import pandas as pd # easy load of csv
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, path_csv, path_root):
        self._path_csv  = path_csv
        self._path_root = path_root
        self.annotations = pd.read_csv(path_csv)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self):
        pass

    def sample(self):
        """
        Sample and display a random frame from the dataset.
        """

        samples = []
        labels  = []
        fig, ax = plt.subplots(1,4, figsize=(15,8))

        for i in range(4):
            rnd_video = np.random.randint(len(self.annotations)) # sample random video
            video_name = self.annotations.iloc[rnd_video,0]
            label      = self.annotations.iloc[rnd_video,1]

            vframes, aframes, info = read_video(self._path_root + video_name, pts_unit='sec')

            sample = vframes[np.random.randint(len(vframes))].numpy()
            
            ax[i].imshow(sample)
            ax[i].set_title(label, fontsize=20, fontweight='bold')


        plt.show()


if __name__ == "__main__":
    path_data = "../data/"
    path_annotations = "../data/annotations.csv"
    dataset = VideoDataset(path_annotations, path_data)

    dataset.sample()