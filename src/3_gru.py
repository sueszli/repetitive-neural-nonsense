from pathlib import Path
import pandas as pd
import polars as pl
import os
import pickle
import sys

module_path = Path(os.getcwd()).parent / "ebnerd-benchmark" / "src"
if module_path not in sys.path:
    sys.path.append(str(module_path))

from ebrec.utils._constants import *
from ebrec.utils._behaviors import create_binary_labels_column, sampling_strategy_wu2019, truncate_history
from ebrec.utils._polars import slice_join_dataframes

"""
load data
"""

data_base = Path(os.getcwd()) / "data-merged" / "merged"
# train_val_base = data_base / "2-ebnerd_small_(80MB)"
# train_val_base = data_base / "3-ebnerd_large_(3.0GB)"
train_val_base = data_base / "1-ebnerd_demo_(20MB)"
test_base = data_base / "5-ebnerd_testset_(1.5GB)"
assert train_val_base.exists() and test_base.exists()

train_behaviors = pl.scan_parquet(train_val_base / "train" / "behaviors.parquet")
train_history = pl.scan_parquet(train_val_base / "train" / "history.parquet")

val_behavior = pl.scan_parquet(train_val_base / "validation" / "behaviors.parquet")
val_history = pl.scan_parquet(train_val_base / "validation" / "history.parquet")

test_behavior = pl.scan_parquet(test_base / "test" / "behaviors.parquet")
test_history = pl.scan_parquet(test_base / "test" / "history.parquet")

train_articles = pl.scan_parquet(train_val_base / "articles.parquet")
val_articles = train_articles
test_articles = pl.scan_parquet(test_base / "articles.parquet")

"""
preprocessing: truncate user history, select subset of columns, join behavior and history, sample based on Wu2019, add binary labels column
"""


def ebnerd_from_path(history: pl.LazyFrame, behaviors: pl.LazyFrame, history_size: int = 30) -> pl.DataFrame:
    df_history = (history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(truncate_history,
                                                                                        column = DEFAULT_HISTORY_ARTICLE_ID_COL,
                                                                                        history_size = history_size,
                                                                                        padding_value = 0))
    df_behaviors = (behaviors.collect().pipe(slice_join_dataframes, df2 = df_history.collect(), on = DEFAULT_USER_COL,
                                             how = "left"))
    return df_behaviors


COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]

HISTORY_SIZE = 2
N_SAMPLES = 1000

df_train = (
    ebnerd_from_path(history = train_history, behaviors = train_behaviors,
                     history_size = HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(sampling_strategy_wu2019, npratio = 4, shuffle = True, with_replacement = True, seed = 123)
    .pipe(create_binary_labels_column)
    .sample(n = N_SAMPLES)
)

df_validation = (
    ebnerd_from_path(history = train_history, behaviors = train_behaviors,
                     history_size = HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(sampling_strategy_wu2019, npratio = 4, shuffle = True, with_replacement = True, seed = 123)
    .pipe(create_binary_labels_column)
    .sample(n = N_SAMPLES)
)

df_test = (
    ebnerd_from_path(history = train_history, behaviors = train_behaviors,
                     history_size = HISTORY_SIZE)
    .select(COLUMNS)
    .with_columns(pl.Series(DEFAULT_CLICKED_ARTICLES_COL, [[]]))
    .pipe(sampling_strategy_wu2019, npratio = 4, shuffle = True, with_replacement = True, seed = 123)
    .pipe(create_binary_labels_column)
    .sample(n = N_SAMPLES)
)

df_train = df_train.to_pandas()
df_validation = df_validation.to_pandas()
df_test = df_test.to_pandas()
df_articles_train = train_articles.collect().to_pandas()
df_articles_val = val_articles.collect().to_pandas()
df_articles_test = test_articles.collect().to_pandas()

"""
helper functions for creating the vocabularies and prepare the format for the sequential iterator
"""


def create_vocabs(train_df, articles_df, user_vocab_file, item_vocab_file, cate_vocab_file):
    user_id_mapping = {value: idx for idx, value in enumerate(train_df['user_id'].unique())}
    item_id_mapping = {value: idx for idx, value in enumerate(articles_df['article_id'].unique())}
    item_category_mapping = {value: idx for idx, value in enumerate(articles_df['category'].unique())}

    with open(user_vocab_file, 'wb') as f:
        pickle.dump(user_id_mapping, f)

    with open(item_vocab_file, 'wb') as f:
        pickle.dump(item_id_mapping, f)

    with open(cate_vocab_file, 'wb') as f:
        pickle.dump(item_category_mapping, f)


def prepare_format(df_train, df_articles, output_file) -> pd.DataFrame:
    default_value = "000000"
    article_category_dict = df_articles.set_index('article_id')['category'].to_dict()
    labels = []
    user_ids = []
    main_article_ids = []
    main_article_categories = []
    timestamps = []
    history_article_ids = []
    history_category_ids = []
    history_timestamps = []

    for index, row in df_train.iterrows():
        uid = row['user_id']
        viewed_articles = row['article_ids_inview']
        click_labels = row['labels']

        accumulated_article_ids = []
        accumulated_article_categories = []
        accumulated_timestamps = []

        for article_id, label in zip(viewed_articles, click_labels):
            category = article_category_dict.get(article_id, "unknown")

            accumulated_article_ids.append(article_id)
            accumulated_article_categories.append(category)
            accumulated_timestamps.append(default_value)

            article_history_str = ",".join(map(str, accumulated_article_ids))
            category_history_str = ",".join(map(str, accumulated_article_categories))
            timestamp_history_str = ",".join(accumulated_timestamps)

            labels.append(label)
            user_ids.append(uid)
            main_article_ids.append(article_id)
            main_article_categories.append(category)
            timestamps.append(default_value)
            history_article_ids.append(article_history_str)
            history_category_ids.append(category_history_str)
            history_timestamps.append(timestamp_history_str)

    df_output = pd.DataFrame({
        'label': labels,
        'user_id': user_ids,
        'article_id': main_article_ids,
        'category_id': main_article_categories,
        'timestamp': timestamps,
        'history_article_ids': history_article_ids,
        'history_category_ids': history_category_ids,
        'history_timestamps': history_timestamps
    })

    df_output.to_csv(output_file, sep='\t', index=False, header=False)

    return df_output

# -----------------------------------------------------------------------------------------------------


from ebrec.models.deeprec.models.sequential.gru import GRUModel
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator

"""
Prepare the data
"""

train_file_path = "train_file.csv"
validation_file_path = "validation_file.csv"
test_file_path = "test_file.csv"
user_vocab_path = 'user_vocab.pkl'
item_vocab_path = 'item_vocab.pkl'
cate_vocab_path = 'cate_vocab.pkl'
intermediate_output = "ebnerd_gru_intermediate.txt"
yaml_file = Path(
    os.getcwd()) / "ebnerd-benchmark-repository" / "src" / "ebrec" / "models" / "deeprec" / "config" / "gru.yaml"
train_num_ngs = 4
valid_num_ngs = 4
test_num_ngs = 9

prepare_format(df_train, df_articles_train, train_file_path)
prepare_format(df_train, df_articles_val, validation_file_path)
prepare_format(df_train, df_articles_test, test_file_path)

train_file = pd.read_csv(train_file_path, sep = '\t', header = None)
validation_file = pd.read_csv(validation_file_path, sep = '\t', header = None)
test_file = pd.read_csv(test_file_path, sep = '\t', header = None)

create_vocabs(df_train, df_articles_train, user_vocab_path, item_vocab_path, cate_vocab_path)

"""
Train model
"""

hparams = prepare_hparams(str(yaml_file),
                          embed_l2 = 0.,
                          layer_l2 = 0.,
                          learning_rate = 0.001,
                          epochs = 1,
                          batch_size = 50,
                          show_step = 20,
                          MODEL_DIR = "model/",
                          SUMMARIES_DIR = "summary/",
                          user_vocab = user_vocab_path,
                          item_vocab = item_vocab_path,
                          cate_vocab = cate_vocab_path,
                          need_sample = True,
                          train_num_ngs = train_num_ngs)

input_creator = SequentialIterator
model = GRUModel(hparams, input_creator, seed = 42)

model.fit(
    train_file = train_file_path,
    valid_file = validation_file_path,
    valid_num_ngs = valid_num_ngs
)

model.predict(test_file_path, intermediate_output)

pred_test = pd.read_csv(intermediate_output)
submission_path = Path(os.getcwd()) / "submissions" / "ebnerd_gru.txt"
with open(submission_path, "w") as f:
    for idx, row in pred_test.iterrows():
        f.write(f"{idx} {row.iloc[0]}\n")

print(model.run_eval(test_file_path, num_ngs = test_num_ngs))




