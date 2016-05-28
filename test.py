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


def run_spectral_experiment(foldername):
    filenames = listdir(foldername)
    data_scores={}
    for filename in filenames:
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        independent_features = [c for c in df.columns.tolist() if "<$" not in c]

        data_scores[filename] = {}
        for nc in xrange(2, int(len(independent_features)**0.5)+1):
            independents = manifold.SpectralEmbedding(n_components=nc, affinity='rbf').fit_transform(df[independent_features])
            dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
            dependent = df[dependent_feature]

            repeats = 20
            auc_scores = []
            for _ in xrange(repeats):
                print ".",
                sys.stdout.flush()

                indexes = range(len(dependent))
                shuffle(indexes)

                shuf_independent = [independents[i] for i in indexes]
                shuf_dependent = [dependent[i] for i in indexes]

                training_size = int(len(shuf_dependent)/2)

                one_independent = shuf_independent[:training_size]
                one_dependent = shuf_dependent[:training_size]

                two_independent = shuf_independent[training_size:]
                two_dependent = shuf_dependent[training_size:]

                cross_folds = [[one_independent, one_dependent, two_independent, two_dependent], [two_independent, two_dependent, one_independent, one_dependent]]
                for indep_train_data, dep_train_data, indep_test_data, dep_test_data in cross_folds:
                    rfc = RandomForestClassifier(n_estimators=300)
                    rfc.fit(indep_train_data, dep_train_data)
                    predictions = rfc.predict(indep_test_data)

                    auc_scores.append(roc_auc_score(dep_test_data, predictions))

            print filename, nc, round(np.median(auc_scores), 3)*100, round(np.std(auc_scores), 3)*100
            data_scores[filename][str(nc)] = [round(np.median(auc_scores), 3)*100, round(np.std(auc_scores), 3)*100]

    pickle.dump(data_scores, open("./Output/SpectralEmbedding/" + foldername.replace("/", "_") + ".p", "wb"))


def run_LLE_experiment(foldername, method):
    filenames = listdir(foldername)
    data_scores={}
    for filename in filenames:
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        independent_features = [c for c in df.columns.tolist() if "<$" not in c]
        print filename

        data_scores[filename] = {}
        for nc in xrange(2, int(len(independent_features)**0.5)+1):
            if int((nc * (nc + 3) / 2) + 1) > 10: nn = int((nc * (nc + 3) / 2)+1)
            else: nn = 10
            independents = manifold.LocallyLinearEmbedding(n_neighbors=nn , n_components=nc, eigen_solver='dense',method=method).fit_transform(df[independent_features])
            dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
            dependent = df[dependent_feature]

            repeats = 20
            auc_scores = []
            for _ in xrange(repeats):
                print ".",
                sys.stdout.flush()

                indexes = range(len(dependent))
                shuffle(indexes)

                shuf_independent = [independents[i] for i in indexes]
                shuf_dependent = [dependent[i] for i in indexes]

                training_size = int(len(shuf_dependent)/2)

                one_independent = shuf_independent[:training_size]
                one_dependent = shuf_dependent[:training_size]

                two_independent = shuf_independent[training_size:]
                two_dependent = shuf_dependent[training_size:]

                cross_folds = [[one_independent, one_dependent, two_independent, two_dependent], [two_independent, two_dependent, one_independent, one_dependent]]
                for indep_train_data, dep_train_data, indep_test_data, dep_test_data in cross_folds:
                    rfc = RandomForestClassifier(n_estimators=300)
                    rfc.fit(indep_train_data, dep_train_data)
                    predictions = rfc.predict(indep_test_data)

                    auc_scores.append(roc_auc_score(dep_test_data, predictions))

            print filename, nc, round(np.median(auc_scores), 3)*100, round(np.std(auc_scores), 3)*100
            data_scores[filename][str(nc)] = [round(np.median(auc_scores), 3)*100, round(np.std(auc_scores), 3)*100]

    pickle.dump(data_scores, open("./Output/LLE_" + method + "/" + foldername.replace("/", "_") + ".p", "wb"))




def run_default_experiment(foldername):
    filenames = listdir(foldername)
    data_scores={}
    for filename in filenames:
        if filename == ".DS_Store": continue
        df = pd.read_csv(foldername + filename)
        independent_features = [c for c in df.columns.tolist() if "<$" not in c]

        independents = df[independent_features]
        dependent_feature = [c for c in df.columns.tolist() if "<$" in c][-1]
        dependent = df[dependent_feature]

        repeats = 20
        auc_scores = []
        for _ in xrange(repeats):
            print ".",
            sys.stdout.flush()

            indexes = range(len(dependent))
            shuffle(indexes)

            shuf_independent = [independents.iloc[i] for i in indexes]
            shuf_dependent = [dependent.iloc[i] for i in indexes]

            training_size = int(len(shuf_dependent)/2)

            one_independent = shuf_independent[:training_size]
            one_dependent = shuf_dependent[:training_size]

            two_independent = shuf_independent[training_size:]
            two_dependent = shuf_dependent[training_size:]

            cross_folds = [[one_independent, one_dependent, two_independent, two_dependent], [two_independent, two_dependent, one_independent, one_dependent]]
            for indep_train_data, dep_train_data, indep_test_data, dep_test_data in cross_folds:
                rfc = RandomForestClassifier(n_estimators=300)
                rfc.fit(indep_train_data, dep_train_data)
                predictions = rfc.predict(indep_test_data)

                auc_scores.append(roc_auc_score(dep_test_data, predictions))

        print filename, round(np.median(auc_scores), 3)*100, round(np.std(auc_scores), 3)*100
        data_scores[filename] = [round(np.median(auc_scores), 3)*100, round(np.std(auc_scores), 3)*100]

    pickle.dump(data_scores, open("./Output/Default/" + foldername.replace("/", "_") + ".p", "wb"))

if __name__ == "__main__":
    for folder_name in folder_names:
        # run_default_experiment(data_folder + folder_name + "/")
        # run_spectral_experiment(data_folder + folder_name + "/")
        methods = [  'modified', 'ltsa', 'standard','hessian',]
        for method in methods:
            run_LLE_experiment(data_folder + folder_name + "/", method)

