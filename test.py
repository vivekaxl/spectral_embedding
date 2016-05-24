from __future__ import division
from os import listdir
import pandas as pd
import numpy as np

data_folder = "./Data/"
normalized_data_folder = "./NData/"
folder_names = ["PROMISE", "AEEEM", "NASA"]

def run_cleaning(foldername):
    filenames = listdir(foldername)
    for filename in filenames:
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
        df[dependent_feature] = df[dependent_feature].apply(lambda x: 1 if x > 0 else 0)
        df.to_csv(foldername + filename, index=False)

def run_normalization(foldername):
    filenames = listdir(foldername)
    for filename in filenames:
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        independent_features = [c for c in df.columns.tolist() if "<$" not in c]
        for independent_feature in independent_features:
            mean = np.mean(df[independent_feature])
            std = np.std(df[independent_feature])
            df[independent_feature] = df[independent_feature].apply(lambda x: round((x-mean)/std, 5))

        df.to_csv(foldername + filename, index=False)
        print filename




if __name__ == "__main__":
    for folder_name in folder_names:
        run_normalization(data_folder + folder_name + "/")