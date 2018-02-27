#!/bin/python3
'''
Generates a directory of the CSV files from the hyperstream for quick future reference, and
includes functions for a variety of other hyperstream summarizations.
'''
import os
from functools import lru_cache

hyperstream_outname = "hyperstream_directory.csv"
usa_outname = "usa_directory.csv"

stream_headers = [
    "id",
    "username",
    "time",
    "message",  # index 3
    "author_ID",
    "is_original",
    "original_author",
    "retweeted_from",
    "notes",
    "comment",
    "tags",
    "location",  # index 11
    "site",
    "language",
    "possibly_sensitive",
    "retweet_count",
    "favorite_count",
    "source",
    "location_dup"
]

stream_types = [  # values for each field
    float,
    str,
    str,
    str,
    float,
    bool,
    str,
    str,
    int,
    str,
    str,
    str,
    str,
    str,
    bool,
    int,
    int,
    str,
    str
]


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


def ntabs(path):
    '''

    :param path:
    :return: list of number of tabs on each line (one tab for each line)
    '''
    with open(path, encoding="utf-8") as f:
        return [l.count("\t") for l in f if l if l != "" and l != "\n"]


@lru_cache(maxsize=1024)
def shape(path):
    '''
    :return: shape of a tsv or csv whose path is passed.
    '''
    return df(path).shape


@lru_cache(maxsize=1)
def df(path, use_hyp_names=False, infer_header=False, use_hyperstream_repair=True):
    import pandas as pd
    global stream_headers

    if use_hyp_names and infer_header:
        raise TypeError("Use of hyperstream column names AND header inference cannot both be set to true.")

    try:
        if path.endswith(".csv"):
            if not infer_header:
                if use_hyp_names is False:
                    return pd.read_csv(path, header=None)
                else:
                    return pd.read_csv(path, header=None, names=stream_headers)
            else:
                return pd.read_csv(path)
        elif path.endswith(".tsv"):
            if not infer_header:
                if use_hyp_names is False:
                    return pd.read_csv(path, delimiter="\t", header=None)
                else:
                    return pd.read_csv(path, delimiter="\t", header=None, names=stream_headers)
            else:
                return pd.read_csv(path, delimiter="\t")
        else:
            raise ValueError("function df passed a file of unknown type. Files must terminate in .csv or .tsv.")
    except pd.errors.ParserError as e:
        if use_hyperstream_repair:
            return repair_hyperstream_tsv(path, names=stream_headers)
        else:
            raise e


@lru_cache(maxsize=2)
def hyperstream_directory(directory=os.getcwd(), update_file=None):
    '''
    assumes stream name does not have the numeral "2" or the period "." (both expected in filename though).

    update_file: path for the last run of this function on this data (saves time of re-calculating)

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
def append_states(path, shapefiledir=os.path.join(os.getcwd(), "tl_2017_us_state"), local_shapefile="tl_2017_us_state"):
    '''
    Gives the US state (or null value) geotagged in the tweet.

    Defaults to provided geocoordinates if they are not equal to (0,0) (twitter's null geocoords). Parses the name of
    the location if geocoords are 0,0. Written for use with 2017 census shapefiles.

    :param path: path of tsv to have state appended to it
    :param shapefiledir: directory where the state shapefiles are located
    :param local_shapefile: name of shapefiles

    :return: dataframe of tweets from passed tsv with state appended as column
    '''
    import shapefile  # pip3 install pyshp
    import shapely.geometry
    global stream_headers

    sf = shapefile.Reader(os.path.join(shapefiledir, local_shapefile))
    shapes = sf.shapes()
    recs = sf.records()

    state_abbreviations = [recs[i][6] for i in range(len(recs))]

    def state_from_name(name):
        s = name.split(", ")
        if len(s) != 2: return None  # format unknown todo review potential other formats
        abbr = s[1].upper()
        i = state_abbreviations.find(abbr)
        return recs[i][7] if i != -1 else None

    def within_state(geostr):
        name, category, lat, long = geostr.split("   ")  # three spaces. vals: name, type (e.g. urban or poi), lat, long

        if float(lat) == 0 and float(long) == 0:
            if category != "poi":
                return state_from_name(name)
            else:
                return None  # todo give support for points of interest. rerun whole usa_directory when supported.

        p = shapely.geometry.Point(float(lat), float(long))
        for i in range(len(shapes)):
            if p.within(shapely.geometry.shape(shapes[i])):
                return recs[i][7]  # 6 is the two-letter code, 7 is the full name
        return None

    dat = df(path, use_hyp_names=True)
    dat['state'] = dat.location.apply(within_state)

    return dat


@lru_cache(maxsize=1)
def usa_directory(directory=os.getcwd(), update_file=None, hyp_dir_file=None):
    import pandas as pd

    if hyp_dir_file is None:
        hd = hyperstream_directory(directory=directory)
    else:
        hd = hyperstream_directory(directory=directory, update_file=hyp_dir_file)
    usa = hd[hd['filename'].str.startswith("USA")]

    def num_and_unique(f):
        return append_states(f).dropna().shape[0], len(append_states(f).state.dropna().unique())

    if update_file is not None:
        l = usa.shape[0]
        old = pd.read_csv(update_file)  # throws FileNotFoundError if passed bad input for update_file
        usa = old.merge(usa, on="path", how="left")
        new = usa[pd.isnull(usa.usa_cases)]
        new['usa_cases'], new['unique_territories'] = pd.zip(*usa.path.apply(num_and_unique))
        usa = usa.merge(new, on="path", how="right")
        assert usa.shape[0] == l
    else:
        usa['usa_cases'], usa['unique_territories'] = pd.zip(*usa.path.apply(num_and_unique))

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


def get_tab_inconsistency(path, num=18, verbose=True):
    with open(path, encoding="utf-8") as f:
        vs = [l for l in f if l.count("\t") != num and l != "" and l != "\n"]
        if verbose: print("Number of bad lines located: " + str(len(vs)) + " in file " + path)
        if verbose: print("Length of each bad line: " + str([len(l) for l in vs]))
        return vs


def assert_consistent_line_length(verbose=True, stop_after=None, writeout=True):
    '''
    tests that every line of every tsv has the same number of tabs.
    '''
    tsvs = files_from_dir()
    prev = None
    if writeout: inconsistencies = open("tab_inconsistency_cases", "w", encoding="utf-8")
    good = True
    bads = []
    found = 0
    for tsv in tsvs:
        n = ntabs(tsv['path'])
        if 1 != len(set(n)):
            print("inconsistent number of tabs in " + tsv['path'])
            for line in get_tab_inconsistency(tsv['path']):
                if writeout: inconsistencies.write(line)
                bads.append(line)
                found += 1
            if stop_after is not None and found >= stop_after:
                if writeout: inconsistencies.close()
                raise AssertionError("Inconsistencies found.")
            good = False
        p = n[0]
        if prev is not None:
            if p != prev:
                print("cross-file inconsistency detected in " + tsv['path'])
                prev = p
                good = False
        else:
            prev = p
        if verbose: print(str(p) + " " + tsv['path'])
    if writeout: inconsistencies.close()
    assert good
    return True


def run_tests():
    assert_old_accurate()
    assert_consistent_colnum()
    assert_consistent_line_length()  # fails. a more interesting test would be to see if the repair function works.
    return True


# todo test this
# def hyperstream_type_check(l_case):
#     '''
#
#     :param l_case: list with same order as headers
#     :return: bool (True if cases pass type tests)
#     '''
#     global stream_headers, stream_types
#
#     if len(l_case) != 18: return False
#
#     return all([stream_types[i](l_case[i]) for i in range(len(l_case)) if l_case[i] != "null"])


def repair_hyperstream_tsv(path, names=stream_headers):
    '''
    two known cases of problems:
    1.) extra tabs in the copy of location
    2.) io error causing two lines to be printed on the same line (sometimes with loss)

    since these errors account for an extreme minority of the total data,
    (one case in millions for either, and apparently only for some points of interest in case 1),
    I'm comfortable removing these data in cases where repairing could introduce inaccuracy to the data.
    '''
    import re

    import pandas as pd

    id_exp = r"[0-9]{18}\t"  # matches with twitter ids.

    exp = re.compile(id_exp)

    def repaired(line):
        if line.count("\t") == 18:  # good
            return tuple(line.split("\t"))
        elif len(exp.findall(line)) == 0:  # likely second line of a case 1 or 2 problem. remove it.
            return None
        elif len(exp.findall(line)) > 1:  # likely case 2 problem. remove it.
            return None
        else:  # likely case 1 problem. replace duplicate of location with uncorrupted original.
            s = line.split("\t")[0:18]
            s.append(s[11])  # duplicate location
            return tuple(s)  # if hyperstream_type_check(s) else None

    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([repaired(l) for l in f if repaired(l) != None], columns=names)


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
