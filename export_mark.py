import os

import pandas as pd

import generate_directory as gd

outdir = "~/to_mark"
locale_df = pd.read_csv(os.path.join(os.path.curdir, os.path.join("mark_in", "bounding_boxes_limited.csv")))
hd = gd.hd()

names = locale_df.subunit.values  # strings
values = locale_df.bbox.values  # strings

for f in hd[hd.filename.str.startswith("mark")].path.values:
    print("Preparing " + f)
    outname = os.path.join(outdir, os.path.basename(f).split(".")[0] + ".csv.gz")
    gd.locations_from_bounds(gd.df(f), names, values).to_csv(outname, index=False, compression="gzip")
    print("File saved to " + outname)
