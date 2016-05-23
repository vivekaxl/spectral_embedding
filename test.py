from __future__ import division
from os import listdir
import pandas as pd

data_folder = "./Data/"
folder_names = ["AEEEM", "NASA", "PROMISE"]

def run_experiment(foldername):
    filenames = listdir(foldername)
    for filename in filenames:
        data = pd.read_csv(filename)


    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    for folder_name in folder_names:
        run_experiment(data_folder + folder_name + "/")