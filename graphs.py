from __future__ import division
from os import listdir
import sys
import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.metrics import roc_auc_score
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
import pickle

data_folder = "./Data/"
normalized_data_folder = "./NData/"
folder_names = ["PROMISE", "AEEEM", "NASA"]
data_folder_names = ["Default", "Experiment"]





if __name__ == "__main__":
    for folder_name in folder_names:
        filename = "._Data_" + folder_name + "_.p"
        default_filename = "./Output/Default/" + filename
        default_data_scores = pickle.load(open(default_filename, "rb"))

        experiment_filename = "./Output/Experiment/" + filename
        experiment_data_scores = pickle.load(open(experiment_filename, "rb"))
        files = experiment_data_scores.keys()
        for file in files:
            # print "filename, default, " + ",".join(map(str, sorted(experiment_data_scores[file].keys())))
            result = [file, str(default_data_scores[file][0])]
            keys = experiment_data_scores[file].keys()
            for key in sorted(keys):
                result.append(str(experiment_data_scores[file][key][0]))
            print ",".join(result)

        print "----" * 30

