#!/bin/python3
'''
Generates a directory of the CSV files from the hyperstream for quick future reference, and
includes functions for a variety of other hyperstream summarizations. Provides various utilities for partitioning data.
'''
import multiprocessing
import os
from functools import lru_cache

import numpy as np
import shapely.geometry  # pip3 install shapely

hyperstream_outname = "hyperstream_directory.csv"
usa_outname = "usa_directory.csv"

threads = multiprocessing.cpu_count() - 2  # conservative. increase to taste.
if threads < 1:
    threads = 1

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

type_list = [  # values for each field
    str,  # really an int but they get bigger than int64s.
    str,
    str,
    str,
    np.float64,
    bool,
    str,
    str,
    np.uint64,
    str,
    str,
    str,
    str,
    str,
    bool,
    np.uint64,
    np.uint64,
    str,
    str
]

stream_types = dict()
for i in range(len(stream_headers)):
    stream_types[stream_headers[i]] = type_list[i]

us_dtype = {"name": str, "category": str, "lat": np.float64, "long": np.float64, "us_state": str}


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
    :return: shape of a tsv or csv whose path is passed. Uses hyperstream_repair by default.
    '''
    return df(path).shape


@lru_cache(maxsize=1)
def df(path):
    import pandas as pd
    import os
    global stream_headers, stream_types

    try:
        if path.endswith(".csv.gz") or path.endswith(".csv"):
            t = {**stream_types, **us_dtype} if os.path.basename(path).upper().startswith("USA") else stream_types
            return pd.read_csv(path, dtype=t, compression="gzip")
        elif path.endswith(".tsv"):
            return pd.read_csv(path, delimiter="\t", header=None, names=stream_headers, dtype=stream_types)
        else:
            raise NotImplementedError(
                "function df passed a file of unknown type. Files must terminate in .csv.gz or .tsv.")
    except (pd.errors.ParserError, ValueError):
        return repair_hyperstream_tsv(path)


def hyperstream_directory(directory=os.getcwd(), update_file=None, verbose=False):
    '''
    assumes stream name does not have the numeral "2" or the period "." (both expected in filename though).

    update_file: path for the last run of this function on this data (saves time of re-calculating those cases).

    '''
    import pandas as pd

    if verbose: print("GENERATING HYPERSTREAM DIRECTORY. Looking for tsv files in " + directory)
    tsvs = files_from_dir(directory=directory, suffix=".tsv")
    if verbose: print("Search complete. " + str(len(tsvs)) + " tsv files found.")
    if update_file is None:
        old = None
    else:
        old = pd.read_csv(update_file)  # throws FileNotFoundError if passed bad input for old_name
        if verbose: print("Old directory file loaded.")
    if verbose:
        i = 0
        new = 0
    for tsv in tsvs:
        if old is None or tsv['updated'] > os.path.getmtime(update_file):
            if verbose:
                new += 1
                print("New file detected (new file # " + str(new) + "). Analyzing.")
            tsv['nrow'] = nrow(tsv['path'])
            tsv['ncol'] = ncol(tsv['path'])
            tsv['topic'] = tsv['filename'].split("2")[0]  # streaming dates start with 20**
            tsv['date'] = tsv['filename'][tsv['filename'].find("2"): tsv['filename'].find(".")]
        else:
            case = old[old.path == tsv['path']].iloc[0].to_dict()
            for k in case.keys():
                tsv[k] = case[k]
        if verbose:
            i += 1
            if (len(tsvs) / i) % 10 == 0:
                print("Ratio of files checked: " + str(i / len(tsvs)))
                print("New files recorded so far: " + str(new))

    if verbose:
        print("hyperstream_directory complete. " + str(new) + \
              " new files recorded (total of " + str(i) + \
              " files recorded, including those in update_file). Converting result to dataframe to be returned.")
    return pd.DataFrame(tsvs)


def append_states(path, shapefiledir=os.path.join(os.getcwd(), "tl_2017_us_state"), local_shapefile="tl_2017_us_state",
                  verbose=0):
    '''
    Gives the US state (or null value) geotagged in the tweet.

    Defaults to provided geocoordinates if they are not equal to (0,0) (twitter's null geocoords). Parses the name of
    the location if geocoords are 0,0. Written for use with 2017 census shapefiles.

    :param path: path of tsv to have state appended to it
    :param shapefiledir: directory where the state shapefiles are located
    :param local_shapefile: name of shapefiles

    :return: dataframe of tweets from passed tsv with state appended as column
    '''
    import numpy as np
    import pandas as pd
    import shapefile  # pip3 install pyshp
    global stream_headers, threads, us_dtype

    sf = shapefile.Reader(os.path.join(shapefiledir, local_shapefile))
    shapes = sf.shapes()
    recs = sf.records()
    shapelies = [shapely.geometry.shape(shapes[i]) for i in range(len(shapes))]
    state_abbreviations = [recs[i][5] for i in range(len(recs))]
    state_names = [recs[i][6].lower() for i in range(len(recs))]

    out = []

    dat = df(path)

    for geostr in dat.location.values:
        try:
            name, category, lat, long = geostr.split(
                "   ")  # three spaces. vals: name, type (e.g. urban or poi), lat, long
        except ValueError:
            print("BAD GEOSTR PASSED TO APPEND_STATES FROM FILE " + path)
            print(geostr)
            out.append({"name": "Unknown", "category": "Unknown", "lat": np.nan, "long": np.nan, "us_state": "Unknown"})
            continue
        try:
            lat, long = float(lat), float(long)
        except ValueError:
            print("BAD LAT / LONG VALUES PASSED TO APPEND_STATES FROM FILE " + path)
            lat, long = 0., 0.

        case = {"name": name, "category": category, "lat": lat, "long": long}

        if lat == 0 and long == 0:
            case['lat'], case['long'] = np.nan, np.nan
            if category != "poi":
                s = name.split(", ")
                if len(s) != 2:
                    case['us_state'] = "Unknown"
                    out.append(case)
                    continue
                    # format unknown todo review potential other formats
                try:
                    v = recs[state_abbreviations.index(s[1].upper())][6]
                    if verbose > 1: print("State found via name. Name: " + name + ". State: " + v)
                    case['us_state'] = v
                    out.append(case)
                    continue
                except ValueError:  # thrown by abbreviations.index(abbreviation) when value not in list
                    try:
                        v = recs[state_names.index(s[0].lower())][6]
                        if verbose > 1: print("State found via name. Name: " + name + ". State: " + v)
                        case['us_state'] = v
                        out.append(case)
                        continue
                    except ValueError:
                        pass  # make sure this pass is always followed by an append / continue
            # todo give support for points of interest. regenerate entire usa_directory when supported.
            case['us_state'] = "Unknown"
            out.append(case)
            continue

        point = shapely.geometry.Point(lat, long)

        for i in range(len(shapelies)):
            if point.within(shapelies[i]):
                if verbose > 1: print("State found via geocoord. State: " + recs[i][6])
                case['us_state'] = recs[i][6]
                out.append(case)
                break
        else:
            case['us_state'] = "Unknown"
            out.append(case)

    return pd.concat([dat, pd.DataFrame(out).astype(us_dtype)], axis=1)


def usa_directory(directory=os.getcwd(), update_file=None, hd=None, hd_file=hyperstream_outname, verbose=0):
    if verbose > 0: print("GENERATING USA DIRECTORY.")
    if update_file is not None:
        raise NotImplementedError
    else:
        gzips = files_from_dir(suffix=".csv.gz")
        for csv in gzips:
            frame = df(csv)
            raise NotImplementedError


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


def get_tab_inconsistency(path, num=18, verbose=True):  # todo turn this into a generator
    with open(path, encoding="utf-8") as f:
        vs = [l for l in f if l.count("\t") != num and l != "" and l != "\n"]
        if verbose: print("Number of bad lines located: " + str(len(vs)) + " in file " + path)
        if verbose: print("Length of each bad line: " + str([len(l) for l in vs]))
        return vs


def assert_consistent_line_length(verbose=True, stop_after=None, writeout=True, outname="tab_inconsistency_cases"):
    '''
    tests that every line of every tsv has the same number of tabs.
    '''
    tsvs = files_from_dir()
    prev = None
    if writeout: inconsistencies = open(outname, "w", encoding="utf-8")
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


def repair_hyperstream_tsv(path, length=19, verbose=True):
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

    global stream_types, stream_headers

    id_exp = r"[0-9]{18}\t"  # matches with twitter ids.

    exp = re.compile(id_exp)

    bads = [0]

    def repaired(line):
        if line.count("\t") == 18:  # good
            return tuple(line.split("\t"))
        elif len(exp.findall(line)) == 0:  # likely second line of a case 1 or 2 problem. remove it.
            bads[0] += 1
            return None
        elif len(exp.findall(line)) > 1:  # likely case 2 problem. remove it.
            bads[0] += 1
            return None
        else:
            try: # likely case 1 problem. replace duplicate of location with uncorrupted original.
                s = line.split("\t")[0:18]
                s.append(s[11])  # duplicate location
                return tuple(s)  # if hyperstream_type_check(s) else None
            except IndexError:
                return None #very rare error wherein there aren't 11 values.

    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame.from_records([repaired(l) for l in f if repaired(l) != None],
                                       columns=stream_headers).astype(stream_types)
        if verbose:
            import os
            print("# misformatted cases excluded from file " + os.path.basename(path) + ": " + str(bads[0]))
        return df


def __query_one_file__(path, directory, regex, date, username, location_type, author_id, language):  # todo untested
    start = "all["
    selection = []
    if regex is not None:
        selection.append("(all.message.str.contains(regex))")
    if date is not None:
        selection.append("(all.date == date)")
    if username is not None:
        selection.append("(all.username.str.lower() == username.lower())")
    if location_type is not None:
        selection.append("(all.location.split("   ")[1] == location_type)")
    if author_id is not None:
        selection.append("(all.author_ID == author_id)")
    if language is not None:
        selection.append("(all.language == language)")
    end = "]"

    all = df(path)
    if len(selection) > 1:
        return eval(start + " & ".join(selection) + end)
    elif selection == 1:
        return eval(start + selection[0][1:-1] + end)
    else:
        return eval(all)


def query(tweet_regex=None, date=None, topic=None, filepaths=None,
          username=None, location_type=None, author_id=None, language=None,
          directory=None):  # todo untested
    # hyperstream_directory(update_file=hyperstream_outname)
    global threads

    if directory is None:
        global hyperstream_directory, hyperstream_outname
        directory = hyperstream_directory(update_file=hyperstream_outname)

    # todo add input validation

    def one_f_params(f):
        return [f, directory, tweet_regex, date, username, location_type, author_id, language]

    def many(paths):
        import multiprocessing
        with multiprocessing.Pool(threads) as p:
            return pd.concat(p.starmap(__query_one_file__, [one_f_params(f) for f in paths]), ignore_index=True)

    if filepaths is None:
        if topic is None:
            return many(directory['path'].values)
        else:
            return many(directory[directory.topic == topic]['path'].values)
    elif topic is not None:
        raise NotImplementedError("Partitioning a set of files based on topic is out of scope for query().")
    elif type(filepaths) == str:
        return __query_one_file__(*one_f_params(filepaths))
    else:
        return many(filepaths)  # assumes iterable


def random_sample(n, directory, seed=1001):  # todo untested
    '''
    :param n: total number of tweets to pull.
    :param directory: directory dataframe (with same columns as hyperstream_directory)
    :param seed: initial seed for the numpy random number generator
    :return:
    '''
    import numpy.random
    import pandas as pd
    rando = numpy.random.RandomState(seed=seed)
    from_each = directory.nrow * n // directory.nrow.sum()  # roughly correct

    def get(i):
        dc = directory.iloc[i]
        one = df(directory.path[i]).sample(n=from_each[i], random_state=rando)
        for v in dc.index:
            one[v] = dc[v]
        return one

    # modify from_each to make sure we get precisely n samples.
    deficit = n - sum(from_each)
    if deficit > 0:
        addis = rando.random_integers(len(from_each) - 1, size=(deficit,))
        for i in addis:
            from_each[i] += 1
    elif deficit < 0:
        subis = rando.random_integers(len(from_each) - 1, size=(abs(deficit),))
        for i in subis:
            from_each[i] -= 1

    assert n == sum(from_each)

    if directory.shape[0] == 1:
        return get(i)
    return pd.concat([get(i) for i in range(len(from_each))], ignore_index=True)


def usa_mig_helper(index, length, filename, path, target):
    print("Beginning migration for file: " + filename + " (" + str(index + 1) + " out of " + str(length) + ").")
    d = append_states(path)
    d[d['us_state'] != "Unknown"].to_csv(
        os.path.join(target, filename.split(".")[0] + ".csv.gz"),
        compression="gzip", index=False)
    print("Migration for file " + filename + " complete.")


def migrate_and_reformat_known_usa(target, usa, multi=False):
    if multi:  # warning: multithreading here slowed down and eventually hung.
        with multiprocessing.Pool(threads) as p:
            p.starmap(usa_mig_helper,
                      [[i, usa.shape[0], usa.filename.iloc[i], usa.path.iloc[i], target] for i in range(usa.shape[0])])
    else:
        for i in range(usa.shape[0]):
            usa_mig_helper(i, usa.shape[0], usa.filename.iloc[i], usa.path.iloc[i], target)


def usa_first_gzips():
    hd = hyperstream_directory(update_file=hyperstream_outname)
    migrate_and_reformat_known_usa("/home/dominic/shiny/hyperstream/usa_gzips", hd[hd.topic == "USA"])


if __name__ == "__main__":
    import pandas as pd

    print("generate_directory running.")

    try:
        hd = hyperstream_directory(update_file=hyperstream_outname, verbose=True)
    except FileNotFoundError:
        print("Invalid update_file passed. Rerunning without update_file.")
        hd = hyperstream_directory(verbose=True)
    hd.to_csv(hyperstream_outname, index=False)

    try:
        ud = usa_directory(update_file=usa_outname, hd=hd, verbose=1)
    except FileNotFoundError:
        print("Invalid update_file passed. Rerunning without update_file.")
        ud = usa_directory(hd=hd, verbose=1)
    ud.to_csv(usa_outname, index=False)
