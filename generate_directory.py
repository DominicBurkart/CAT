'''
Generates a directory of the CSV files from the hyperstream for quick future reference.

Does not perform content checking.
'''
import os
from functools import lru_cache

outname = "hyperstream_directory.csv"


def files_from_dir(directory=os.getcwd(), suffix=".tsv"):
    '''
    returns all files with the given suffix in the given directory or its children.
    :param directory:
    :param suffix:
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


def nrows(path):
    '''
    number of rows in a file.
    '''
    with open(path, encoding="utf-8") as f:
        return len([1 for l in f if l])


@lru_cache()
def hyperstream_directory(directory=os.getcwd(), old_name=None):
    '''
    assumes name of stream does not have the numeral "2" or the period "." (both expected in filename though).
    '''
    import pandas as pd

    tsvs = files_from_dir(directory=directory)
    if old_name is not None:
        old = pd.read_csv(old_name)
        for tsv in tsvs:
            if tsv['updated'] > os.path.getmtime(old_name):
                tsv['len'] = nrows(tsv['path']) - 1  # subtract one because the EOF is on its own line
                tsv['topic'] = tsv['filename'].split("2")[0]  # streaming dates start with 20**
                tsv['date'] = tsv['filename'][tsv['filename'].find("2"): tsv['filename'].find(".")]
            else:
                case = old[old.path == tsv['path']].iloc[0]
                tsv['len'] = case['len']
                tsv['topic'] = case['topic']
                tsv['date'] = case['date']
    else:
        for tsv in tsvs:
            tsv['len'] = nrows(tsv['path']) - 1  # subtract one because the EOF is on its own line
            tsv['topic'] = tsv['filename'].split("2")[0]  # streaming dates start with 20**
            tsv['date'] = tsv['filename'][tsv['filename'].find("2"): tsv['filename'].find(".")]
    return pd.DataFrame(tsvs)


def assert_old_accurate():
    global outname
    assert hyperstream_directory().equals(hyperstream_directory(old_name=outname))
    return True

if __name__ == "__main__":
    import pandas as pd

    try:
        hyperstream_directory(old_name=outname).to_csv(outname, index=False)
    except FileNotFoundError:
        hyperstream_directory().to_csv(outname, index=False)
