"""
This script converts the json files with graph data from PosePrediction to ndarrays. These
ndarrays serve as input to ST-GCN. The ndarray data is (C, T, V, M) where

    C = 3       (['x', 'y', 'score'])
    T = 300     (no frames)
    V = 18      (no of nodes in human skeleton)
    M = 1       (no humans in each frame)

according to https://arxiv.org/pdf/1801.07455.pdf


The output directory receives the following format:

<output_dir>/
    annotations.csv
    data/
        <label1>/
            <file>.npy
            ...
        <label2>/
            <file>.npy
            ...

"""

from os import listdir
from os.path import join, isfile
import sys, getopt, json
import numpy as np


def parse_args(argv):
    try:
        opts, _ = getopt.getopt(argv, 'hi:o:', ['input_dir=', 'output_dir='])
    except getopt.GetoptError:
       print('json_to_ndarray.py --input_dir <inputdir> --out_dir <outdir>')
       sys.exit(2)

    input_dir = ""
    output_dir = ""
    for opt, arg in opts:
        if opt == '-h':
            print('json_to_ndarray.py -i <inputdir> -o <outdir>')
            sys.exit()
        elif opt in ("-i", "--input_dir"):
            input_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
    return input_dir, output_dir 

def json_to_ndarry(input_dir):
    
    C = 3
    T = 300
    V = 18
    M = 1
    data_numpy = np.zeros((C, T, V, M))

    input_filenames = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]

    for name in input_filenames:
        print("Processing {}...".format(name))
        with open(name) as f:
            data = json.load(f)

        frames, metadata = data["frames"], data["metadata"]

        for frame in frames:
            bodies = frame["bodies"]
            frame_id = frame["frame_id"]
            for body in bodies: # Should only be one human
                body_parts = body["body_parts"]
                for part_id, part_data in body_parts.items():
                    data_numpy[0, frame_id, int(part_id), 0] = part_data["x"]
                    data_numpy[1, frame_id, int(part_id), 0] = part_data["y"]
                    data_numpy[2, frame_id, int(part_id), 0] = part_data["score"]
    
    #TODO: Annotations file

    return data_numpy

def save(path, data):
    pass


if __name__ == "__main__":
    input_dir, output_dir = parse_args(sys.argv[1:])
    data = json_to_ndarry(input_dir)