import os

unlabeled_data_path = "../../ChatExport_27_04_2020/video_files/unlabeled/"
labels = ["snatch", "clean", "backsquat", "other"]

for f in os.listdir(unlabeled_data_path):
    vid_name = unlabeled_data_path + f
    
    print(f)

    os.system("xdg-open {}".format(vid_name))
    exit()