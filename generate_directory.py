'''
Generates a directory of the CSV files from the hyperstream for quick future reference, and
includes functions for a variety of other hyperstream summarizations.
'''
import os
from functools import lru_cache

hyperstream_outname = "hyperstream_directory.csv"
usa_outname = "usa_directory.csv"


@lru_cache(maxsize=1)
def files_from_dir(directory=os.getcwd(), suffix=".tsv"):
    '''
    returns all files with the given suffix in the given directory or its children.

    caches the most recent call and returns it if called with the same parameters.

    :param directory: string of directory to search.
    :param suffix: includes only files with names that end with this string
    :return: list of dictionaries with the keys 'folder', 'filename', and 'path'
    '''
    out = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(suffix):
                out.append({
                    'folder': dirpath,
                    'filename': f,
                    'path': os.path.join(dirpath, f),
                    'updated': os.path.getmtime(os.path.join(dirpath, f))
                })
    return out


@lru_cache(maxsize=1024)
def nrow(path):
    '''
    number of rows in a file, or lines in a csv/tsv.
    '''
    try:
        return shape(path)[0]
    except NotImplementedError:
        with open(path, encoding="utf-8") as f:
            return len([1 for l in f if l])


def ncol(path):
    '''
    number of columns in a csv or tsv.
    '''
    return shape(path)[1]


@lru_cache(maxsize=1024)
def shape(path):
    '''
    :return: shape of a tsv or csv whose path is passed.
    '''
    return df(path).shape


def df(path):
    import pandas as pd
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, delimiter="\t")
    else:
        raise NotImplementedError("function df passed a file of unknown type. Files must terminate in .csv or .tsv.")


@lru_cache(maxsize=2)
def hyperstream_directory(directory=os.getcwd(), update_file=None):
    '''
    assumes stream name does not have the numeral "2" or the period "." (both expected in filename though).

    update_file: path for an early

    '''
    import pandas as pd

    tsvs = files_from_dir(directory=directory, suffix=".tsv")
    if update_file is None:
        old = None
    else:
        old = pd.read_csv(update_file)  # throws FileNotFoundError if passed bad input for old_name
    for tsv in tsvs:
        if old is None or tsv['updated'] > os.path.getmtime(update_file):
            tsv['nrow'] = nrow(tsv['path'])
            tsv['ncol'] = ncol(tsv['path'])
            tsv['topic'] = tsv['filename'].split("2")[0]  # streaming dates start with 20**
            tsv['date'] = tsv['filename'][tsv['filename'].find("2"): tsv['filename'].find(".")]
        else:
            case = old[old.path == tsv['path']].iloc[0].to_dict()
            for k in case.keys():
                tsv[k] = case[k]

    return pd.DataFrame(tsvs)


@lru_cache(maxsize=1)
def append_states(path):
    # import shapefile #pip3 install pyshp
    # import pandas as pd
    raise NotImplementedError()


@lru_cache(maxsize=1)
def usa_directory(directory=os.getcwd(), update_file=None, hyp_dir_file=None):
    if hyp_dir_file is None:
        hd = hyperstream_directory(directory=directory)
    else:
        hd = hyperstream_directory(directory=directory, update_file=hyp_dir_file)
    usa = hd[hd['filename'].str.startswith("USA")]

    nc = lambda f: append_states(f).dropna().shape[0]
    uc = lambda f: len(append_states(f).dropna().unique())

    if update_file is not None:
        old = pd.read_csv(update_file)  # throws FileNotFoundError if passed bad input for update_file
        usa = old.merge(usa, on="path", how="left")
        new = usa[pd.isnull(usa.usa_cases)]
        new['usa_cases'] = new.filename.apply(nc)
        new['unique_territories'] = new.filename.apply(uc)
        usa = usa.merge(new, on="path", how="right")
    else:
        usa['usa_cases'] = usa.filename.apply(nc)
        usa['unique_territories'] = usa.filename.apply(uc)

    return usa


def assert_old_accurate():
    '''
    tests that hyperstream directory yields consistent results when updating vs generating directory csv.
    '''
    global hyperstream_outname
    assert hyperstream_directory(update_file=hyperstream_outname).equals(hyperstream_directory())
    return True


def assert_consistent_colnum():
    '''
    tests that all hyperstream output files have the same number of columns
    '''
    assert len(hyperstream_directory().ncol.unique()) == 1
    return True


def run_tests():
    assert_old_accurate()
    assert_consistent_colnum()
    return True


if __name__ == "__main__":
    import pandas as pd

    try:
        hyperstream_directory(update_file=hyperstream_outname).to_csv(hyperstream_outname, index=False)
    except FileNotFoundError:
        hyperstream_directory().to_csv(hyperstream_outname, index=False)

    try:
        usa_directory(update_file=usa_outname, hyp_dir_file=hyperstream_outname).to_csv(usa_outname, index=False)
    except FileNotFoundError:
        usa_directory(hyp_dir_file=hyperstream_outname).to_csv(usa_outname, index=False)
