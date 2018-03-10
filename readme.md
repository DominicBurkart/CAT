# Classification of Autobiographical text

CAT is the first project using data collected from [hyperstream] (https://github.com/DominicBurkart/SocialNetworkAnalysis/blob/master/Applications/Hyperstream1.java).

## Contents

`generate_directory.py` is a simple script used to collate and query data output
from `hyperstream1`, providing a light and convenient pipeline for dealing with
large amounts of data in flat files.

## Project overview

CAT is currently in progress. Below is an overview of the next steps:

1. Complete `usa_directory` in `generate_directory.py`.
2. Complete `random_sample` in `generate_directory.py`.
3. Collect a random sample of 5,000 USA-geotagged english tweets for rating.
Rate them, and generate the the relevant classifiers. Use cross-validation with
20% of the data (1,000 tweets). Save trained models as pickles in this repo.
4. Augment `random_sample` to accept exclusion criteria (minimum: tweet IDs).
5. Use `random_sample` to collect 1,000 tweets from other topic streams of
interest. Rate them and compare classifier accuracy at different confidence
quantiles (find correlation between classifier confidence and accuracy).
6. Assuming that the correlation between classifier confidence and accuracy is
high and that CAT is a useful diagnostic, the project will be considered a
success and will be considered complete. Improvements to the accuracy and
generalizability of classifier will be considered as needed.
