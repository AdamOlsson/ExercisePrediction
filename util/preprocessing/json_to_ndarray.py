"""
This script converts the json files with graph data from PosePrediction to ndarrays. These
ndarrays serve as input to ST-GCN.
"""

import sys, getopt


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

def json_to_ndarry(input_dir, output_dir):
    pass

if __name__ == "__main__":
    input_dir, output_dir = parse_args(sys.argv[1:])
    json_to_ndarry(input_dir, output_dir)