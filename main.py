from Datasets.GraphDataset import GraphDataset

dataset = GraphDataset(save_dir="../datasets/weightlifting/graphs", raw_dir="PosePrediction/data/graphs/videos")

print(dataset.raw_file_names)
print(dataset.processed_file_names)