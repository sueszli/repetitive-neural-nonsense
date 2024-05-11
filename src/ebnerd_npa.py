from pathlib import Path
import tensorflow as tf
import polars as pl
import numpy as np
import os

from ebrec.utils._constants import *
from ebrec.utils._behaviors import create_binary_labels_column, sampling_strategy_wu2019, add_known_user_column, add_prediction_scores, truncate_history
from ebrec.utils._articles import convert_text2encoding_with_transformers, create_article_id_to_value_mapping
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._nlp import get_transformers_word_embeddings

from transformers import AutoTokenizer, AutoModel


"""
load data
"""

data_base = Path(os.getcwd()) / "data-merged" / "merged"
# train_val_base = data_base / "1-ebnerd_demo_(20MB)"
train_val_base = data_base / "2-ebnerd_small_(80MB)"
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

articles_word2vec: pl.LazyFrame = pl.scan_parquet(data_base / "7-Ekstra-Bladet-word2vec_(133MB)" / "document_vector.parquet")
articles_image_embeddings: pl.LazyFrame = pl.scan_parquet(data_base / "8-Ekstra_Bladet_image_embeddings_(372MB)" / "image_embeddings.parquet")
articles_contrastive_vector: pl.LazyFrame = pl.scan_parquet(data_base / "9-Ekstra-Bladet-contrastive_vector_(341MB)" / "contrastive_vector.parquet")
articles_bert_base_multilingual_cased: pl.LazyFrame = pl.scan_parquet(data_base / "10-google-bert-base-multilingual-cased_(344MB)" / "bert_base_multilingual_cased.parquet")
articles_xlm_roberta_base: pl.LazyFrame = pl.scan_parquet(data_base / "11-FacebookAI-xlm-roberta-base_(341MB)" / "xlm_roberta_base.parquet")


"""
preprocessing: truncate user history, select subset of columns, join behavior and history, sample based on Wu2019, add binary labels column
"""


def ebnerd_from_path(history: pl.LazyFrame, behaviors: pl.LazyFrame, history_size: int = 30) -> pl.DataFrame:
    df_history = history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(truncate_history, column=DEFAULT_HISTORY_ARTICLE_ID_COL, history_size=history_size, padding_value=0)
    df_behaviors = behaviors.collect().pipe(slice_join_dataframes, df2=df_history.collect(), on=DEFAULT_USER_COL, how="left")
    return df_behaviors


COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
]
HISTORY_SIZE = 30
N_SAMPLES = 100
df_train = (
    ebnerd_from_path(history=train_history, behaviors=train_behaviors, history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(sampling_strategy_wu2019, npratio=4, shuffle=True, with_replacement=True, seed=123)
    .pipe(create_binary_labels_column)
    .sample(n=N_SAMPLES)
)
df_validation = ebnerd_from_path(history=val_history, behaviors=val_behavior, history_size=HISTORY_SIZE).select(COLUMNS).pipe(create_binary_labels_column).sample(n=N_SAMPLES)
df_test = (
    ebnerd_from_path(history=test_history, behaviors=val_behavior, history_size=HISTORY_SIZE)
    .with_columns(pl.Series(DEFAULT_CLICKED_ARTICLES_COL, [[]]))
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
    .sample(n=N_SAMPLES)
)


"""
use huggingface transformers to convert article text to tokens, tokens to embeddings
"""

df_articles = train_articles.collect()
TRANSFORMER_MODEL_NAME = "bert-base-multilingual-cased"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)


# -----------------------------------------------------------------------------------------------------

"""
batch data
"""
from ebrec.models.newsrec.dataloader import LSTURDataLoader  # NPA and LSTUR share the same dataloader

train_dataloader = LSTURDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=64,
)
val_dataloader = LSTURDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=32,
)
test_dataloader = LSTURDataLoader(
    behaviors=df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=32,
)


"""
train model
"""
from ebrec.models.newsrec.model_config import hparams_npa
from ebrec.models.newsrec.npa import NPAModel

MODEL_NAME = "NPA"
LOG_DIR = f"./runs/{MODEL_NAME}"
MODEL_WEIGHTS = f"./runs/data/state_dict/{MODEL_NAME}/weights"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
modelcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, verbose=1)

config = hparams_npa
model = NPAModel(hparams=config, word2vec_embedding=word2vec_embedding, seed=42)
model.model.summary()
hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=1,
    callbacks=[tensorboard_callback, early_stopping_callback, modelcheckpoint_callback],
)
model.model.load_weights(filepath=MODEL_WEIGHTS)
