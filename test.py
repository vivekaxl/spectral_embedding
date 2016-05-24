from __future__ import division
from os import listdir
import pandas as pd

data_folder = "./Data/"
folder_names = ["PROMISE", "AEEEM", "NASA",]

def run_cleaning(foldername):
    filenames = listdir(foldername)
    for filename in filenames:
        df = pd.read_csv(foldername + filename)
        dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
        df[dependent_feature] = df[dependent_feature].apply(lambda x: 1 if x > 0 else 0)



        import pdb
        pdb.set_trace()



    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    for folder_name in folder_names:
        run_cleaning(data_folder + folder_name + "/")