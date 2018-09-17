import os
import pickle
import re
import time

import fasttext
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

data_file = "encoded_5000_usa.csv"

use_sbatch = False
verbose = False

NPERMS = 10

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


def hypt(accuracy, label, text, perms=NPERMS, show_graph=True, name="hyptest", print_progress=True,
         multiprocess=True, save_perm_accuracies=True):
    '''
    Tests whether classifiers are performing significantly better than chance.
    Permutation-based hypothesis testing based on removing the correspondence between the IV and the DV via
    randomization to generate a null distribution.
    :param accuracy:
    :param label:
    :param text:
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
                civ = copy.deepcopy(label)
                np.random.shuffle(civ)
                resps.append(
                    pool.apply_async(five_fold,
                                     args=(civ, text),
                                     kwds=async_kwargs,
                                     callback=out_dicts.append))
            for r in resps:
                r.wait()
            null_accuracy = [d['accuracy'] for d in out_dicts]
    else:
        for i in range(perms):
            civ = copy.deepcopy(label)
            np.random.shuffle(civ)
            null_accuracy.append(five_fold(civ, text, show_graph=False, hyp_test=False, write_out=False)['accuracy'])
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


def five_fold(label, text, name=None, show_feat=False, hyp_test=True, show_graph=False, write_out=True):
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
    full = list(zip(label, text))
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
            out = {"accuracy": a, "null probability": hypt(a, label, text, show_graph=show_graph)}
        else:
            out = {"accuracy": a,
                   "null probability": hypt(a, label, text, name=name, show_graph=show_graph),
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


def o(value, outname):
    n_bytes = 2 ** 31  # the pickle package is slightly broken in python 3.4 and requires these params for saving large files.
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(value)
    with open(outname, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])
    return True


def save_nb(label, text, name="5000_naive_bayes.pickle"):
    train = getFeatureSet(list(zip(label, text)))
    model = nltk.classify.scikitlearn.SklearnClassifier(MultinomialNB()).train(train)
    return o(model, name)


def parse_for_fasttext(datafile):
    df = pd.read_csv(datafile)
    outname = "temporary_parse_for_fasttext" + str(time.time())
    with open(outname) as out:
        for label, text in zip(df.autobiographical, df.message):
            if label == "0":
                out.write("__label__nonautobiographical__ " + text + "\n")
            else:
                out.write("__label__autobiographical__ " + text + "\n")
    return outname


def fasttext_five_fold(datafile):
    file = parse_for_fasttext(datafile)
    results = []
    with open(file).readlines() as lines:
        for traini, testi in KFold(n_splits=5).split(lines):
            with open("train.txt", "w") as train:
                train.writelines([lines[i] for i in traini])
            with open("test.txt", "w") as test:
                test.writelines([lines[i] for i in testi])
            m = fasttext.supervised("train.txt", "fasttext_five_fold")
            results.append(m.test("test.txt").precision)
    for f in [file, "test.txt", "test.txt", "fasttext_five_fold.bin", "fasttext_five_fold.vec"]:
        os.remove(f)
    return {"mean": np.mean(results), "all": results}


def fasttext_perm_test(datafile, make_plot=False):
    '''
    :param datafile:
    :return: float. probability that the real precision of the classification model is greater than chance.
    '''
    df = pd.read_csv(datafile)
    out = dict()
    null_precision = []
    for i in range(NPERMS):
        df.autobiographical = np.random.shuffle(df.autobiographical)
        df.to_csv("disordered.csv", index=False)
        null_precision.append(fasttext_five_fold("disordered.csv")['mean'])
    os.remove("disordered.csv")

    d = fasttext_five_fold(datafile)
    real_avg = d['mean']
    out['null probability'] = float(len([v for v in null_precision if v > real_avg])) / len(null_precision)
    out['accuracy'] = real_avg
    out['name'] = "fasttext on " + str(datafile)
    if make_plot:
        import plotly.graph_objs as go
        from plotly.offline import plot

        fig = go.Figure(data=[go.Histogram(x=d['all'], opacity=0.9)])
        plot(fig, filename="fasttext_fivefold_accuracies.html")

        fig = go.Figure(data=[go.Histogram(x=null_precision, opacity=0.9)],
                        layout={"shapes":
                            [{
                                'type': 'line',
                                'xref': 'x',
                                'yref': 'y',
                                'x0': d['mean'],
                                'y0': 0,
                                'x1': d['mean'],
                                'y1': 5000,
                                'line': {
                                    'color': 'rgb(50, 171, 96)',
                                    'width': 2,
                                }
                            }]
                        })
        plot(fig, filename="fasttext_fivefold_permutations.html")


def save_fasttext(datafile, name="5000_fasttext"):
    f = parse_for_fasttext(datafile)
    fasttext.skipgram(f, name)
    os.remove(f)


def rcnn_perm_test(labels, text):
    raise NotImplementedError


def save_rcnn(labels, text, name="5000_rcnn.pickle"):
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(13)
    df = pd.read_csv(data_file)
    save_nb(df.autobiographical, df.message)
    save_fasttext(data_file)
    save_rcnn(df.autobiographical, df.message)
    models = [
        five_fold(df.autobiographical, df.message, "cat_usa_fivefold"),
        fasttext_perm_test(data_file, make_plot=True),
        rcnn_perm_test(df.autobiographical, df.message)
    ]
    pd.DataFrame.from_dict(models).to_csv("cat_classifier_accuracies.csv", index=False)
