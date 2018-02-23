'''
Generates a directory of the CSV files from the hyperstream for quick future reference.

Does not perform content checking.
'''
import os

from functools import lru_cache


def files_from_dir(directory=os.getcwd(), suffix=".tsv"):
    '''
    returns all files with the given suffix in the given directory or its children.
    :param directory:
    :param suffix:
    :return: list of dictionaries with the keys 'folder', 'filename', and 'path'
    '''
    (dirpath, dirnames, filenames) = os.walk(directory)
    out = []
    for f in filenames:
        if f.endswith(suffix):
            out.append({
                'folder': dirpath,
                'filename': f,
                'path': os.join(dirpath, f)
            })
    for d in dirnames:
        out.extend(files_from_dir(directory=d))
    return out


def nrows(path):
    '''
    number of rows in a file.
    '''
    with open(path) as f:
        return len([1 for l in f])


@lru_cache()
def hyperstream_directory(directory=os.getcwd()):
    tsvs = files_from_dir(directory=directory)
    for tsv in tsvs:
        tsv['len'] = nrows(tsv['path'])
        tsv['topic'] = tsv['f'].split("_")


if __name__ == "__main__":
    hyperstream_directory()
