#!/bin/python3
'''
Generates a directory of the CSV files from the hyperstream for quick future reference, and
includes functions for a variety of other hyperstream summarizations. Provides various utilities for partitioning data.

Encodes null integer values as -999.

If run regularly, these commands should be relatively quick – In cases where there are no new files, this script takes
1.68 seconds to complete while scanning large directories on a hard drive.

Potential site of error: if files are uploaded to the directory after hyperstream_directory has begun, they won't be
included in that hyperstream_directory. This is a non-issue for our work, as this program a.) is not run while files
are being written to the relevant directory and b.) is run often enough that the next time will catch it.
'''
import multiprocessing
import os
from functools import lru_cache

import langdetect  # pip3 install langdetect
import numpy as np
import shapefile  # pip3 install pyshp
import shapely.geometry  # pip3 install shapely

hyperstream_outname = "hyperstream_directory.csv"
usa_outname = "usa_directory.csv"
gzip_outname = "gzip_directory.csv"

usa_path = os.path.abspath(os.path.join(os.curdir, "usa_gzips"))

threads = max([multiprocessing.cpu_count() - 1, 2])

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
    str,  # id is really an int but they get bigger than int64s.
    str,
    str,
    str,
    np.int64,
    bool,
    str,
    str,
    np.int64,
    str,
    str,
    str,
    str,
    str,
    bool,
    np.int64,
    np.int64,
    str,
    str
]

stream_types = dict()
for i in range(len(stream_headers)):
    stream_types[stream_headers[i]] = type_list[i]

us_dtype = {"name": str, "category": str, "lat": np.float64, "long": np.float64, "us_state": str}


@lru_cache(maxsize=1)
def files_from_dir(directory=os.getcwd(), suffix=".tsv", just=None):
    '''
    returns all files with the given suffix in the given directory or its children.

    caches the most recent call and returns it if called with the same parameters.

    :param directory: string of directory to search.
    :param suffix: includes only files with names that end with this string. ignored if param just is not None.
    :just: list of basenames. only return information about these files in the given directory. Asserts all are found.
    :return: list of dictionaries with the keys 'folder', 'filename', and 'path'
    '''
    out = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(suffix) or (just is not None and f in just):
                out.append({
                    'folder': dirpath,
                    'filename': f,
                    'path': os.path.join(dirpath, f),
                    'updated': os.path.getmtime(os.path.join(dirpath, f))
                })
    if just is not None: assert len(just) == len(out)
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
            return len([1 for l in f if l not in ["", "\n"]])


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

    try:
        if path.endswith(".csv.gz") or path.endswith(".csv"):
            t = {**stream_types, **us_dtype} if os.path.basename(path).upper().startswith("USA") else stream_types
            if path.endswith(".gz"):
                return pd.read_csv(path, dtype=t, compression="gzip")
            return pd.read_csv(path, dtype=t)
        elif path.endswith(".tsv"):
            return pd.read_csv(path, delimiter="\t", header=None, names=stream_headers, dtype=stream_types)
        else:
            raise NotImplementedError(
                "function df passed a file of unknown type. Files must terminate in .csv.gz or .tsv.")
    except (pd.errors.ParserError, ValueError, OverflowError) as e:
        if path.endswith(".tsv"):
            return repair_hyperstream_tsv(path)
        else:
            raise NotImplementedError("parsererror / valueerror while attempting to open zipped csv. File: " +
                                      os.path.basename(path) +
                                      "\nerror: " + str(e) +
                                      "\nfull path: " + path)


def gzip_hd(directory=os.getcwd(), update_file=None, verbose=False):
    '''
    Generates a hyperstream directory from gzipped CSVs instead of uncompressed TSVs (what hyperstream_directory does).
    '''
    import pandas as pd

    if verbose: print("GENERATING HYPERSTREAM DIRECTORY. Looking for .csv.gz files in " + directory)
    data_files = files_from_dir(directory=directory, suffix=".csv.gz")
    if update_file is None:
        old = None
    else:
        old = pd.read_csv(update_file)  # throws FileNotFoundError if passed bad input for old_name
        if verbose: print("Old directory file loaded.")

    if old is not None and all(csv['filename'] in old.filename.values for csv in data_files):
        if verbose: print("No new files since update_file was saved. Returning dataframe from update_file.")
        return old

    if verbose:
        i = 0
        newi = 0
    for csv in data_files:
        if old is None or csv['filename'] not in old.filename.values:
            if verbose:
                print("New file detected (new file # " + str(newi + 1) + "). Analyzing.")
            try:
                csv['nrow'] = nrow(csv['path'])  # df is opened and cached.
                csv['ncol'] = ncol(csv['path'])
                csv['topic'] = csv['filename'].split("2")[0]  # streaming dates start with 20**
                csv['date'] = csv['filename'][csv['filename'].find("2"): csv['filename'].find(".")]
                if verbose:
                    newi += 1
            except (ValueError, MemoryError): # df changed while being opened.
                if verbose:
                    print("Excluding file as it produced a memory or value error (perhaps it was changed while reading?): " +
                          str(csv['path']) +
                          "\nDecrementing new file #. This program will attempt to process the file " +
                          "the next time it is run.")
        else:
            case = old[old.path == csv['path']].iloc[0].to_dict()
            for k in case.keys():
                csv[k] = case[k]
        if verbose:
            i += 1
            if (len(data_files) / i) % 10 == 0:
                print("Ratio of files checked: " + str(i / len(data_files)))
                print("New files recorded so far: " + str(newi))

    if verbose:
        print("hyperstream_directory complete. " + str(newi) + \
              " new files recorded (total of " + str(i) + \
              " files recorded, including those in update_file). Converting result to dataframe to be returned.")

    new = pd.DataFrame(data_files)
    if old is not None: assert old[~old.path.isin(new.path)].shape[0] == 0  # check for missing data.
    return new


def hd():
    return hyperstream_directory(update_file=hyperstream_outname)

def hyperstream_directory(directory=os.getcwd(), update_file=None, verbose=False, gzips=False):
    '''


    assumes stream name does not have the numeral "2" or the period "." (both expected in filename though).

    :param directory: directory of the datafiles we need metadata for.
    :param update_file: last hyperstream file
    :param verbose:
    :param gzips: if True, just passes params to gzip_hd and returns the result.
    :return:
    '''
    import pandas as pd

    if gzips: return gzip_hd(directory, update_file, verbose)

    if verbose: print("GENERATING HYPERSTREAM DIRECTORY. Looking for tsv files in " + directory)
    tsvs = files_from_dir(directory=directory, suffix=".tsv")
    if verbose: print("Search complete. " + str(len(tsvs)) + " tsv files found.")
    if update_file is None:
        old = None
    else:
        old = pd.read_csv(update_file)  # throws FileNotFoundError if passed bad input for old_name
        if verbose: print("Old directory file loaded.")

    if old is not None and all(tsv['filename'] in old.filename.values for tsv in tsvs):
        if verbose: print("No new files since update_file was saved. Returning dataframe from update_file.")
        return old

    if verbose:
        i = 0
        newi = 0
    for tsv in tsvs:
        if old is None or tsv['filename'] not in old.filename.values:
            if verbose:
                print("New file detected (new file # " + str(newi + 1) + "). Analyzing.")
            try:
                tsv['nrow'] = nrow(tsv['path'])  # df is opened and cached.
                tsv['ncol'] = ncol(tsv['path'])
                tsv['topic'] = tsv['filename'].split("2")[0]  # streaming dates start with 20**
                tsv['date'] = tsv['filename'][tsv['filename'].find("2"): tsv['filename'].find(".")]
                if verbose:
                    newi += 1
            except (ValueError, MemoryError): # df changed while being opened.
                if verbose:
                    print("Excluding file as it produced a memory or value error (perhaps it was changed while reading?): " +
                          str(tsv['path']) +
                          "\nDecrementing new file #. This program will attempt to process the file " +
                          "the next time it is run.")
        else:
            case = old[old.path == tsv['path']].iloc[0].to_dict()
            for k in case.keys():
                tsv[k] = case[k]
        if verbose:
            i += 1
            if (len(tsvs) / i) % 10 == 0:
                print("Ratio of files checked: " + str(i / len(tsvs)))
                print("New files recorded so far: " + str(newi))

    if verbose:
        print("hyperstream_directory complete. " + str(newi) + \
              " new files recorded (total of " + str(i) + \
              " files recorded, including those in update_file). Converting result to dataframe to be returned.")

    new = pd.DataFrame(tsvs)
    if old is not None: assert old[~old.path.isin(new.path)].shape[0] == 0  # check for missing data.
    return new


def unpack_location(geostr):
    try:
        name, category, lat, long = geostr.split("   ")  # three spaces. vals: name, type (e.g. urban or poi), lat, long
    except ValueError:
        name, category, lat, long = "Unknown", "Unknown", np.nan, np.nan
        return name, category, lat, long
    try:
        lat, long = np.float64(lat), np.float64(long)
    except ValueError:
        lat, long = np.nan, np.nan

    return name, category, lat, long


def apply_and_concat(dataframe, field, func, column_names):
    '''
    from Dennis Golomazov at https://stackoverflow.com/questions/23690284/
    pandas-apply-function-that-returns-multiple-values-to-rows-in-pandas-dataframe
    '''
    import pandas as pd
    return pd.concat((
        dataframe,
        dataframe[field].apply(lambda cell: pd.Series(func(cell), index=column_names))), axis=1)


def unpack_location_df(df):
    return apply_and_concat(df, 'location', unpack_location, column_names=['location_name', 'category', 'lat', 'long'])


def locations_from_bounds(frame, bound_names, bound_vals):
    '''
    Selects the first matching bounding box for each case.

    :param frame:
    :param bound_names:
    :param bound_vals:
    :return:
    '''
    from shapely.geometry.point import Point

    def split_vals(i, length=4):
        if type(bound_vals[i]) != str:
            if len(bound_vals[i]) == length:
                return bound_vals[i]
        if bound_vals[i].startswith("(") and bound_vals[i].endswith(")"):
            return bound_vals[i][1: -1].split(",")
        else:
            return bound_vals[i].split(",")

    def locate(points):
        for tup in points:
            if any(v is np.nan or type(v) not in [np.float64, float] for v in tup):
                yield "Unknown"
                continue

            point = Point(*tup)
            for i in range(len(boxes)):
                if boxes[i].contains(point):
                    yield bound_names[i]
                    break
            else:
                yield "Unknown"

    assert len(bound_names) == len(bound_vals)

    boxes = []
    for i in range(len(bound_vals)):
        boxes.append(
            shapely.geometry.box(*[np.float64(v) for v in split_vals(i)]))  # breaks if bound_vals is misformatted.

    unpacked = unpack_location_df(frame)
    return unpacked.assign(bound_name=list(locate(zip(unpacked.lat.values, unpacked.long.values))))


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


def usa_directory(directory=usa_path, update_file=usa_outname, hd=None, verbose=0):
    import pandas as pd
    if verbose > 0: print("GENERATING USA DIRECTORY.")

    def it(gzips):
        for gzip in gzips:
            try:
                gzip['nrow'] = nrow(gzip['path'])
                gzip['ncol'] = ncol(gzip['path'])
                gzip['topic'] = "USA"
                gzip['date'] = gzip['filename'][gzip['filename'].find("2"): gzip['filename'].find(".")]
            except (ValueError, MemoryError):
                if verbose > 0:
                    print("Excluding file as it produced a memory or value error (perhaps it was changed while reading?): " +
                          str(gzip['path']) +
                          "\nThis program will attempt to process the file " +
                          "the next time it is run.")

        return pd.DataFrame(gzips)

    if update_file is not None:
        if hd is None: raise NotImplementedError("usa_directory needs a hyperstream_directory to function.")
        try:
            old = pd.read_csv(update_file)
        except FileNotFoundError:
            print("USA_UPDATE FILE NOT FOUND: " + str(update_file))
            print("RERUNNING WITHOUT UPDATE FILE. VERIFY YOU MEANT TO CALL THIS SCRIPT FROM THIS DIRECTORY.")
            return usa_directory(directory=directory, update_file=None, hd=hd, verbose=verbose)
        olds = [f.split(".")[0] + ".tsv" for f in old.filename]
        potentials = hd[hd.topic == "USA"].filename
        to_migrate = hd[hd.filename.isin(p for p in potentials if p not in olds)]
        migrate_and_reformat_known_usa(directory, to_migrate, verbose=True if verbose > 1 else False)

        # ok! so now we have our old usa_directory and our mixed old/new data.

        all_gzips = files_from_dir(directory, suffix=".csv.gz")
        new_gzips = [d for d in all_gzips if d['path'] not in old.path.values]
        assert old[~old.path.isin(d['path'] for d in all_gzips)].shape[0] == 0  # checks that no data has gone missing
        if verbose > 0: print("USA directory generated. Returning.")
        return old.append(it(new_gzips), ignore_index=True) if len(new_gzips) > 0 else old

    if hd is None:
        print("No hyperstream directory passed to usa_directory. Generating and returning " + \
              "a usa_directory as best as possible.")
        return it(files_from_dir(directory, suffix=".csv.gz"))

    all_gzips = files_from_dir(directory, suffix=".csv.gz")

    to_migrate = hd[(hd.topic == "USA") &
                    ~(hd.filename.isin(f['filename'].split(".")[0] + ".tsv" for f in all_gzips))]
    migrate_and_reformat_known_usa(directory, to_migrate, verbose=True if verbose > 1 else False)
    new_gzips = all_gzips + files_from_dir(directory,
                                           just=tuple(f.split(".")[0] + ".csv.gz" for f in to_migrate.filename))
    if verbose > 0: print("USA directory generated. Returning.")
    return it(new_gzips)


def assert_old_accurate():
    '''
    tests that hyperstream directory yields consistent results when updating vs generating directory csv.
    '''
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
    assert_consistent_line_length()  # fails. a more useful test would be to see if the repair function works.
    return True


def repair_hyperstream_tsv(path, verbose=True):
    '''
    two known cases of problems:
    1.) extra tabs in the copy of location
    2.) io error causing two lines to be printed on the same line (sometimes with loss)

    since these errors account for an extreme minority of the total data,
    (one case in millions for either, and apparently only for some points of interest in case 1),
    I'm comfortable removing these data in cases where repairing could introduce inaccuracy to the data.
    '''
    import re
    import numpy as np
    import pandas as pd

    id_exp = r"[0-9]{18}\t"  # matches with twitter ids.

    exp = re.compile(id_exp)

    bads = [0]

    def dcheck(l):
        '''
        Types each variable in the passed tuple according to the types in type_list.
        :param l: list passed from repaired() or None
        :return: tuple of typed values.
        '''
        if l is None: return None
        if len(l) != 19:
            bads[0] += 1
            return None
        out = []
        for i in range(len(l)):
            try:
                out.append(type_list[i](l[i]))
            except ValueError:
                if type_list[i] == np.int64 and l[i] is None or type(l[i]) == np.nan:
                    out.append(np.int64(-999))  # encode null values as -999
                else:
                    try:
                        print("Unusual error while dchecking. Excluding case: " + str(l))
                    except ValueError or TypeError:
                        print("Unusual error while dchecking. Case is unprintable. Excluding.")
                    bads[0] += 1
                    return None
        return tuple(out)

    def repaired(line):
        if line.count("\t") == 18:  # good
            return line.split("\t")
        elif len(exp.findall(line)) == 0:  # likely second line of a case 1 or 2 problem. remove it.
            bads[0] += 1
            return None
        elif len(exp.findall(line)) > 1:  # likely case 2 problem. remove it.
            bads[0] += 1
            return None
        else:
            try:  # likely case 1 problem. replace duplicate of location with uncorrupted original.
                s = line.split("\t")[0:18]
                s.append(s[11])  # duplicate location
                return s
            except IndexError:
                return None

    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame.from_records([dcheck(repaired(l)) for l in f if dcheck(repaired(l)) != None],
                                       columns=stream_headers)
        if verbose:
            import os
            print("# misformatted cases excluded from file " + os.path.basename(path) + ": " + str(bads[0]))
        return df.astype(stream_types)


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


def recalc_lang(text, seed=1001):
    langdetect.DetectorFactory.seed = seed
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return "und"


def query(tweet_regex=None, date=None, topic=None, filepaths=None,
          username=None, location_type=None, author_id=None, language=None,
          directory=None):  # todo untested
    import pandas as pd
    # hyperstream_directory(update_file=hyperstream_outname)

    if directory is None:
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


def random_sample_helper(i, verbose, directory, from_each, disclude_tweet_ids, rando):
    if verbose:
        print("Getting values from " + directory.filename.iloc[i] +
              " (file " + str(i + 1) + " of " + str(len(from_each)) + ")")
    dc = directory.iloc[i]
    frame = df(directory.path.iloc[i])
    if disclude_tweet_ids is not None: frame = frame[~frame.id.isin(disclude_tweet_ids)]
    samp = frame.sample(n=from_each[i], random_state=rando)
    for v in dc.index:
        samp[v] = dc[v]
    return samp


def random_sample(directory, n, seed=1001, verbose=True, multi=False, disclude_tweet_ids=None):
    '''

    Samples roughly the same number of tweets from each directory listing. Slow but functional.

    :param directory: directory dataframe (with at least the same columns as hyperstream_directory)
    :param n: total number of tweets to pull.
    :param seed: initial seed for the numpy random number generator
    :return:
    '''

    # todo implement disclude_tweet_ids
    import numpy.random
    import pandas as pd
    rando = numpy.random.RandomState(seed=seed)
    from_each = (directory.nrow.values * n) // directory.nrow.sum()

    # modify from_each to make sure we get precisely n samples.
    deficit = n - sum(from_each)
    if deficit > 0:  # we need more values to get to n
        addis = rando.randint(len(from_each) - 1, size=(deficit,))
        for i in addis:
            from_each[i] += 1
    elif deficit < 0:  # surplus.
        valids = [i for i in range(len(from_each)) if from_each[i] > 0]
        subis = rando.randint(valids, size=(abs(deficit),))
        for i in subis:
            from_each[valids[i]] -= 1

    from_each = tuple(from_each)
    assert n == sum(from_each)

    if directory.shape[0] == 1:
        return random_sample_helper(i, verbose, directory, from_each, rando)
    if not multi:
        return pd.concat([random_sample_helper(i, verbose, directory, from_each, disclude_tweet_ids, rando) for i in
                          range(len(from_each))],
                         ignore_index=True)
    with multiprocessing.Pool(threads) as p:
        return pd.concat(
            p.starmap(random_sample_helper,
                      ((i, verbose, directory, from_each, disclude_tweet_ids, rando) for i in range(len(from_each)))),
            ignore_index=True)


usa_mig_sum = [0]


def usa_mig_helper(index, length, filename, path, target, verbose):
    if verbose:
        print("Beginning migration for file: " + filename + " (" + str(index + 1) + " out of " + str(length) + ").")
    d = append_states(path)
    out = d[d['us_state'] != "Unknown"]
    out.to_csv(
        os.path.join(target, filename.split(".")[0] + ".csv.gz"),
        compression="gzip", index=False)
    if verbose: print("Migration for file " + filename + " complete. Number of tweets: " + str(out.shape[0]))
    usa_mig_sum[0] += out.shape[0]


def migrate_and_reformat_known_usa(target, usa, multi=False, verbose=False):
    if multi:  # warning: sometimes this multiprocess hangs and it's never been faster for me (Dominic March 2018).
        with multiprocessing.Pool(threads) as p:
            p.starmap(usa_mig_helper,
                      ((i, usa.shape[0], usa.filename.iloc[i], usa.path.iloc[i], target, verbose) for i in
                       range(usa.shape[0])))
    else:
        for i in range(usa.shape[0]):
            try:
                usa_mig_helper(i, usa.shape[0], usa.filename.iloc[i], usa.path.iloc[i], target, verbose)
            except MemoryError:
                try:
                    usa_mig_helper(i, usa.shape[0], usa.filename.iloc[i], usa.path.iloc[i], target, verbose)
                except MemoryError:
                    if verbose:
                        print("Excluding "+usa.filename.iloc[i]+" due to MemoryError.")
    if verbose: print("total number of tweets from known US states migrated: " + str(usa_mig_sum[0]))


def usa_first_gzips():
    hd = hyperstream_directory(update_file=hyperstream_outname)
    migrate_and_reformat_known_usa("/home/dominic/shiny/hyperstream/usa_gzips", hd[hd.topic == "USA"], verbose=True)


everything_mig_sum = [0]


def check_target(target, hd):
    target_csvs = [c['filename'].split(".")[0] for c in files_from_dir(target, suffix=".csv.gz")]
    b = hd.filename.str.split(".").str[0]  # david this syntax is bad
    return hd[~b.isin(target_csvs)].path.values


def migrate_everything(target, verbose=True):
    import os
    try:
        hd = hyperstream_directory(update_file=hyperstream_outname, verbose=True)
    except FileNotFoundError:
        hd = hyperstream_directory(verbose=True)

    to_migrate = check_target(target, hd)
    for i in range(len(to_migrate)):
        p = to_migrate[i]
        if verbose: print("Migrating " + os.path.basename(p) + " (" + str(i) + " out of " + str(len(to_migrate)) + ").")
        d = df(p)
        d.to_csv(os.path.join(target, os.path.basename(p).split(".")[0] + ".csv.gz"), compression="gzip", index=False)
        if verbose:
            print(os.path.basename(p) + " migration complete. Number of tweets migrated: " + str(d.shape[0]))
            everything_mig_sum[0] += d.shape[0]
    if verbose:
        print("Migration complete. Total tweets migrated: " + str(everything_mig_sum[0]))


def df_iter(hd):
    for p in hd.path.values:
        yield df(p)

if __name__ == "__main__":
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

