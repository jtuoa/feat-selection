import numpy as np
import argparse
import os
import pdb

parser = argparse.ArgumentParser(description="Feature selection main")
parser.add_argument("--path", type=str, default="", help="path with train_feat.npy and test_feat.npy")
parser.add_argument("--fs", type=str, default="", help="feature selection method")
args = parser.parse_args()

def load_data(filename):
    path = args.path
    data = np.load(os.path.join(path,filename)).item()
    return (data['data'],data['labels'])

def main():
    train_x,train_y = load_data('train_feat.npy')
    test_x,test_y = load_data('test_feat.npy')

main()
