import os

root              = "exercises"
path_data         = "../../data/{}/".format(root) # path relative this file
path_csv_save_loc = "../../data/annotations.csv"  # path relative this file

# entries = []
with open(path_csv_save_loc,'r+') as f:
    data = f.read()
    f.seek(0)
    f.write("# filename,label\n") # Header
    for dir in os.listdir(path_data):
        for file in os.listdir(path_data + dir):
            f.write("{}/{}/{},{}\n".format(root, dir, file, dir)) # path, label
    f.truncate()


