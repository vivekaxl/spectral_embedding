from __future__ import division
from os import listdir
import sys
import pandas as pd
import numpy as np
from sklearn import manifold, datasets
from sklearn.metrics import roc_auc_score
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data_folder = "./Data/"
normalized_data_folder = "./NData/"
folder_names = ["PROMISE", "AEEEM", "NASA"]

def visualize_spectral(foldername, function, save_folder):

    filenames = listdir(foldername)
    for filename in filenames:
        print filename
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        independent_features = [c for c in df.columns.tolist() if "<$" not in c]
        independent = df[independent_features]

        se = function(n_components=2,n_neighbors=10,  affinity='rbf').fit_transform(independent)
        dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
        dependent = df[dependent_feature]
        plt.scatter(se[:, 0], se[:, 1], c=dependent, cmap=plt.cm.Spectral)
        plt.savefig(save_folder + foldername.replace("/", "_") + filename + ".png")
        plt.cla()


def visualize_spectral_3d(foldername, function, save_folder):

    filenames = listdir(foldername)
    for filename in filenames:
        print filename
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        independent_features = [c for c in df.columns.tolist() if "<$" not in c]
        independent = df[independent_features]

        se = function(n_components=3,n_neighbors=20).fit_transform(independent)
        dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
        dependent = df[dependent_feature]
        ax.scatter(se[:, 0], se[:, 1], se[:, 2], c=dependent, cmap=plt.cm.Spectral)
        plt.show()


def visualize_lle(foldername, function, save_folder, method):

    filenames = listdir(foldername)
    for filename in filenames:
        print filename
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        independent_features = [c for c in df.columns.tolist() if "<$" not in c]
        independent = df[independent_features]

        se = function(n_components=2,n_neighbors=10, method=method, eigen_solver='dense').fit_transform(independent)
        dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
        dependent = df[dependent_feature]
        plt.scatter(se[:, 0], se[:, 1], c=dependent, cmap=plt.cm.Spectral)
        plt.savefig(save_folder + foldername.replace("/", "_") + filename + ".png")
        plt.cla()


def visualize_lle_3d(foldername, function, save_folder, method):

    filenames = listdir(foldername)
    for filename in filenames:
        print filename
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        independent_features = [c for c in df.columns.tolist() if "<$" not in c]
        independent = df[independent_features]

        se = function(n_components=3,n_neighbors=10, method=method, eigen_solver='dense').fit_transform(independent)
        dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
        dependent = df[dependent_feature]
        ax.scatter(se[:, 0], se[:, 1], se[:, 2], c=dependent, cmap=plt.cm.Spectral)
        plt.show()


for folder_name in folder_names:
    # visualize_spectral(data_folder + folder_name + "/", manifold.SpectralEmbedding, "./Visuals/Spectral_rbf/")
    # visualize_spectral_3d(data_folder + folder_name + "/", manifold.SpectralEmbedding, "./Visuals/Spectral_rbf/")
    # visualize_lle(data_folder + folder_name + "/", manifold.LocallyLinearEmbedding, "./Visuals/LLE_standard/", 'standard')
    visualize_lle_3d(data_folder + folder_name + "/", manifold.LocallyLinearEmbedding, "./Visuals/LLE_ltsa/", 'ltsa')
    visualize_lle_3d(data_folder + folder_name + "/", manifold.LocallyLinearEmbedding, "./Visuals/LLE_modified/", 'modified')
    visualize_lle_3d(data_folder + folder_name + "/", manifold.LocallyLinearEmbedding, "./Visuals/LLE_hessian/", 'hessian')