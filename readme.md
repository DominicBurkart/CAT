# Classification of Autobiographical Text

CAT is the first project using data collected from <a href="https://github.com/DominicBurkart/SocialNetworkAnalysis/blob/master/Applications/Hyperstream1.java" target="_blank">hyperstream</a>.

Autobiographical text is defined as such in this project:
1. The text discloses emotions or inner thoughts.
2. The emotions or thoughts are not projected onto their source/target.
For example, "Fred made me sad" is autobiographical while "Fred is mean"
is not.
3. As implied by (2), emotions or thoughts are presented in the
first person. They belong to the author or a group the author is a part
of ("Me" or "Us").

## Contents

`generate_directory.py` allows collation and queries to the data collected
from `hyperstream1`, providing a light and convenient pipeline for dealing with
large amounts of data in flat files (either tsv or csv.gz formats).

`get_5000_usa.py` samples five thousand english tweets geotagged with
known-US locations, doing so randomly within-day but weighting how many
tweets from each day based on the total number of tweets collected.

`classifier.py` is used to build and test the models for classifying
text.

## Project Overview

CAT is currently in progress. Below is an overview of the next steps:

1. Complete `usa_directory` in `generate_directory.py`: DONE
2. Complete `random_sample` in `generate_directory.py`: DONE
3. Collect a random sample of 5,000 USA-geotagged english tweets for rating.
Rate them, and generate the relevant classifiers. Use cross-validation with
20% of the data (1,000 tweets). Save trained models as pickles in this repo: READY TO CLASSIFY WHEN HUMAN ENCODING IS COMPLETE.
4. Augment `random_sample` to accept exclusion criteria (minimum: tweet IDs): DONE
5. Use `random_sample` to collect 1,000 tweets from other topic streams of
interest. Rate them and compare classifier accuracy at different confidence
quantiles (find correlation between classifier confidence and accuracy).
6. Assuming that the correlation between classifier confidence and accuracy is
high and that CAT is a useful diagnostic, the project will be considered a
success and will be considered complete. Improvements to the accuracy and
generalizability of classifier will be considered as needed.
