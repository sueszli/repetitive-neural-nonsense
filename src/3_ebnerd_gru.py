from pathlib import Path
import tensorflow as tf
import polars as pl
import numpy as np
import os
import pickle
import logging

import sys

module_path = Path(os.getcwd()).parent / "ebnerd-benchmark" / "src"
if module_path not in sys.path:
    sys.path.append(str(module_path))

from ebrec.utils._constants import *
from ebrec.utils._behaviors import create_binary_labels_column, sampling_strategy_wu2019, add_known_user_column, \
    add_prediction_scores, truncate_history
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes


"""
load data
"""

data_base = Path(os.getcwd()) / "data-merged" / "merged"
train_val_base = data_base / "1-ebnerd_demo_(20MB)"
# train_val_base = data_base / "2-ebnerd_small_(80MB)"
# train_val_base = data_base / "3-ebnerd_large_(3.0GB)"
test_base = data_base / "5-ebnerd_testset_(1.5GB)"
assert train_val_base.exists() and test_base.exists()

train_behaviors = pl.scan_parquet(train_val_base / "train" / "behaviors.parquet")
train_history = pl.scan_parquet(train_val_base / "train" / "history.parquet")

val_behavior = pl.scan_parquet(train_val_base / "validation" / "behaviors.parquet")
val_history = pl.scan_parquet(train_val_base / "validation" / "history.parquet")

test_behavior = pl.scan_parquet(test_base / "test" / "behaviors.parquet")
test_history = pl.scan_parquet(test_base / "test" / "history.parquet")

train_articles: pl.LazyFrame = pl.scan_parquet(train_val_base / "articles.parquet")
val_articles: pl.LazyFrame = train_articles
test_articles: pl.LazyFrame = pl.scan_parquet(test_base / "articles.parquet")

"""
preprocessing: truncate user history, select subset of columns, join behavior, history and articles
"""


def ebnerd_from_path(history: pl.LazyFrame, behaviors: pl.LazyFrame, articles: pl.LazyFrame,
                     history_size: int = 30) -> pl.DataFrame:
    df_history = history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(truncate_history,
                                                                                       column = DEFAULT_HISTORY_ARTICLE_ID_COL,
                                                                                       history_size = history_size,
                                                                                       padding_value = 0)
    df_merged = behaviors.collect().pipe(slice_join_dataframes, df2 = df_history.collect(), on = DEFAULT_USER_COL,
                                            how = "left")

    # df_behaviors_with_articles = df_merged.pipe(slice_join_dataframes, df2 = articles.collect(), on = DEFAULT_ARTICLE_ID_COL, how = 'left')
    # return df_behaviors_with_articles

    return df_merged


COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    #DEFAULT_CATEGORY_COL
]

HISTORY_SIZE = 30
N_SAMPLES = 100

df_train = (
    ebnerd_from_path(history = train_history, behaviors = train_behaviors, articles = train_articles,
                     history_size = HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(sampling_strategy_wu2019, npratio = 4, shuffle = True, with_replacement = True, seed = 123)
    .pipe(create_binary_labels_column)
    .sample(n = N_SAMPLES)
)
df_validation = ebnerd_from_path(history = val_history, behaviors = val_behavior, articles = train_articles,
                                 history_size = HISTORY_SIZE).select(
    COLUMNS).pipe(create_binary_labels_column).sample(n = N_SAMPLES)
df_test = (
    ebnerd_from_path(history = test_history, behaviors = val_behavior, articles = train_articles,
                     history_size = HISTORY_SIZE)
    .with_columns(pl.Series(DEFAULT_CLICKED_ARTICLES_COL, [[]]))
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
    .sample(n = N_SAMPLES)
)

print(" train", df_train)
# print(df_validation)
# print(df_test)

def _create_vocab(train_df, user_vocab, item_vocab, cate_vocab):
    user_dict = {}
    item_dict = {}
    cat_dict = {}

    for index, row in train_df.iter_rows():
        uid = row[0]  # User id
        mid = row[1]  # Article id
        mid_list = row[2]  # Clicked articles
        cat = row[3]  # Category id
        cat_list = row[6]  # don't have that information

        if uid not in user_dict:
            user_dict[uid] = 0
        user_dict[uid] += 1
        if mid not in item_dict:
            item_dict[mid] = 0
        item_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(","):
            if m not in item_dict:
                item_dict[m] = 0
            item_dict[m] += 1

    sorted_user_dict = sorted(user_dict.items(), key = lambda x: x[1], reverse = True)
    sorted_item_dict = sorted(item_dict.items(), key = lambda x: x[1], reverse = True)
    sorted_cat_dict = sorted(cat_dict.items(), key = lambda x: x[1], reverse = True)

    uid_voc = {key: index for index, (key, value) in enumerate(sorted_user_dict)}
    mid_voc = {"default_mid": 0}
    mid_voc.update({key: index + 1 for index, (key, value) in enumerate(sorted_item_dict)})
    cat_voc = {"default_cat": 0}
    cat_voc.update({key: index + 1 for index, (key, value) in enumerate(sorted_cat_dict)})

    with open(user_vocab, "wb") as f:
        pickle.dump(uid_voc, f)
    with open(item_vocab, "wb") as f:
        pickle.dump(mid_voc, f)
    with open(cate_vocab, "wb") as f:
        pickle.dump(cat_voc, f)


from ebrec.models.deeprec.deeprec_utils import prepare_hparams
from ebrec.models.deeprec.models.sequential.gru import GRUModel
from ebrec.models.deeprec.io.sequential_iterator import SequentialIterator

yaml_file = Path(
    os.getcwd()) / "ebnerd-benchmark-repository" / "src" / "ebrec" / "models" / "deeprec" / "config" / "gru.yaml"
valid_num_ngs = 4  # number of negative instances with a positive instance for validation
test_num_ngs = 9  # number of negative instances with a positive instance for testing
output_file = "ebnerd_gru.txt"

user_vocab = "user_vocab.pkl"
item_vocab = "item_vocab.pkl"
cate_vocab = "cate_vocab.pkl"

_create_vocab(train_df = df_train, user_vocab = user_vocab, item_vocab = item_vocab, cate_vocab = cate_vocab)

hparams = prepare_hparams(str(yaml_file))
input_creator = SequentialIterator
model = GRUModel(hparams, input_creator, seed = 42)

hist = model.fit(
    train_file = df_train,
    valid_file = df_validation,
    valid_num_ngs = valid_num_ngs,
    # eval_metric = "group_auc",
)

res_syn = model.run_eval(df_test, num_ngs = test_num_ngs)
print(res_syn)
model = model.predict(df_test, output_file)

logger = logging.getLogger(__name__)
