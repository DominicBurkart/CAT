import pandas as pd

import generate_directory as gd

collect_int = 5000

output_file = "5000_usa_english_sample.csv"

hd = gd.hyperstream_directory(update_file=gd.hyperstream_outname)
ud = gd.usa_directory(hd=hd, update_file=gd.usa_outname)


def r(n, collected=None, m=2, exclude=None):
    '''

    :param n: number of tweets to return
    :param collected: previously collected tweets from this query (for recursive calls).
    :param m: multiplier of number of tweets to ask for total from random_sample (sampled down if too many are valid).
    :param exclude: dataframe with tweet id column (.id)
    :return:
    '''
    disclude = collected.id.values if collected is not None else None
    if exclude is not None:
        disclude = list(disclude) + list(exclude.id.values) if disclude is not None else exclude.id.values


    unchecked = gd.random_sample(ud, collect_int * m, disclude_tweet_ids=disclude)
    goods = unchecked[(unchecked.language == "en") | (unchecked.message.apply(gd.recalc_lang) == "en")]
    # todo confirm twitter lang format here

    if collected is not None:
        goods = pd.concat([collected, goods], ignore_index=True)
        # bad recursive algorithm, but comparatively trivial since sampling takes so long

    if goods.shape[0] > collect_int:
        return goods.sample(collect_int, random_state=1001)
    elif goods.shape[0] == collect_int:
        return goods
    else:  # (goods.shape[0] < collect_int)
        return r(goods.shape[0] - collect_int, collected=goods, m=m + 1, exclude = exclude)


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        s = r(collect_int)
    else:
        s = r(collect_int, exclude = pd.read_csv(sys.argv[1]))
    s['time_sampled_from_stores'] = time.time()
    s.to_csv(output_file, index=False)
