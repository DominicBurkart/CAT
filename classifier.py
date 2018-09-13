import re

import nltk
import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

data_file = "encoded_5000_usa.csv"

use_sbatch = False
verbose = False


def getFeatureSet(twoples):
    """
    @param twoples: the text / categorization from which the word featureset should be extracted.
    The format should look like [("Text with words", "categorization of text"), ("another text", "same/another category")]
    """
    out = []  # list of dictionary/label tuples
    for t in twoples:
        d = dict()
        text = t[0]
        if type(text) != str:
            if verbose:
                print("Non-string passed. Value: " + str(text))
                print("Ignoring value.")
        else:
            text = re.sub(r'\\x[A-Fa-f0-9]+', ' ',
                          text)  # removes characters transcribed from unicode to ascii (e.g. emoji)
            text = re.sub("[^a-zA-Z\d\s']", ' ', text)  # only keeps alphanumeric characters
            words = text.lower().split()
            for w in words:
                if w in d:
                    d[w] += 1
                else:
                    d[w] = 1
            out.append((d, t[1]))  # appends a tuple with the dictionary object and the category to the out list
    return out  # we have our feature set! send it out.


def hypt(accuracy, iv, dv, perms=10000, show_graph=True, name="hyptest", print_progress=True,
         multiprocess=True, save_perm_accuracies=True):
    '''
    Tests whether classifiers are performing significantly better than chance.
    Permutation-based hypothesis testing based on removing the correspondence between the IV and the DV via
    randomization to generate a null distribution.
    :param accuracy:
    :param iv:
    :param dv:
    :param perms:
    :param show_graph:
    :param plot_name:
    :param print_progress:
    :return:
    '''
    import copy
    null_accuracy = []
    if multiprocess:
        import multiprocessing
        async_kwargs = {"hyp_test": False,
                        "show_graph": False,
                        "write_out": False}
        print("Instantiating multiprocessing for " + str(perms) + " permutations on " +
              str(multiprocessing.cpu_count()) + " cores.")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            out_dicts = []
            resps = []
            for i in range(perms):
                civ = copy.deepcopy(iv)
                np.random.shuffle(civ)
                resps.append(
                    pool.apply_async(five_fold,
                                     args=(civ, dv),
                                     kwds=async_kwargs,
                                     callback=out_dicts.append))
            for r in resps:
                r.wait()
            null_accuracy = [d['accuracy'] for d in out_dicts]
    else:
        for i in range(perms):
            civ = copy.deepcopy(iv)
            np.random.shuffle(civ)
            null_accuracy.append(five_fold(civ, dv, show_graph=False, hyp_test=False, write_out=False)['accuracy'])
            if print_progress:
                print("Permutation test iteration #: " + str(i + 1))
                print("Percent complete: " + str(((i + 1) / perms) * 100) + "%")
    if show_graph:
        import plotly.graph_objs as go
        from plotly.offline import plot
        fig = go.Figure(data=[go.Histogram(x=null_accuracy, opacity=0.9)])
        plot(fig, filename=name + ".html")
    g = [s for s in null_accuracy if s >= accuracy]
    if save_perm_accuracies:
        import csv
        with open(name + '_null_accuracies.csv', "w") as f:
            w = csv.writer(f)
            for a in null_accuracy:
                w.writerow([a])
    return len(g) / len(null_accuracy)  # probability of null hypothesis


def five_fold(iv, dv, name=None, show_feat=False, hyp_test=True, show_graph=False, write_out=True):
    '''
    Five-fold cross-validated classification.
    :param iv:
    :param dv:
    :param name:
    :param show_feat:
    :param hyp_test: set True for a permutation hypothesis test that classification is above chance.
    :param show_graph:
    :param write_out:
    :return:
    '''
    full = list(zip(iv, dv))
    v = []  # filled with the average accuracies on left-out data across models. doesn't wait for differences in splits.

    for traini, testi in KFold(n_splits=5).split(full):
        train = (getFeatureSet(full[i]) for i in traini)
        test = (getFeatureSet(full[i]) for i in testi)
        cl = nltk.classify.scikitlearn.SklearnClassifier(MultinomialNB()).train(train)
        v.extend(np.mean((1 if cl.classify(test[i][0]) == test[i][1] else 0 for i in range(len(test)))))
        if show_feat:
            cl.show_most_informative_features(n=100)

    a = np.mean(v)
    if hyp_test:
        if name is None:
            out = {"accuracy": a, "null probability": hypt(a, iv, dv, show_graph=show_graph)}
        else:
            out = {"accuracy": a,
                   "null probability": hypt(a, iv, dv, name=name, show_graph=show_graph),
                   "name": name}
    else:
        if name is None:
            out = {"accuracy": a}
        else:
            out = {"accuracy": a, "name": name}
    if write_out:
        if name is None:
            print("Name for this leave one out analysis is not defined. Data may be overwritten.")
            pd.DataFrame([out]).to_csv("classification_writeout.csv")
        else:
            pd.DataFrame([out]).to_csv(name + ".csv")
    return out


if __name__ == "__main__":
    df = pd.read_csv(data_file)

    if use_sbatch:
        import sbatch

        sbatch.load("five_fold", [df.text, df.aut, "cat_usa_fivefold"])
        sbatch.launch()
    else:
        print(five_fold(df.text, df.aut, "cat_usa_fivefold"))
