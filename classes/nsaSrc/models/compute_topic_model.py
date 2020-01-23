# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import string
import os

from src.models import WordLevelStatistics

from src.models import (
            enrich_significant_terms,
            display_topics,
            topic_exemplars,
            message_topics
)

import pandas as pd
import numpy as np
import pymagnitude

try:
    import umap
    print("Using: umap")
except ImportError:
    print("Using: bhtsne")
    import bhtsne

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

punctuations = string.punctuation
punctuations += '《》–'
stopwords = ["'s"]


data_dir = os.getenv('PWD') + '/data/'

language_model = {'ko': data_dir + '/external/cc.ko.300.magnitude',
                  'en': data_dir + '/external/wiki-news-300d-1M.magnitude'}

target_dim = 10
cluster_selection_method = 'leaf'

text_fn = {'en': data_dir + '/processed/en_flat.txt',
           'ko': data_dir + '/processed/ko_flat.txt'}

sent_id = 0
sentences = []
for sent in open(os.getenv('PWD') + '/data/processed/en_flat.txt'):
    sentences.append({'sent_id': sent_id, 'sentence': sent})
    sent_id += 1


@click.command()
@click.argument('tmx_file', type=click.Path(exists=True))
@click.argument('lang')
@click.option('--percentile', default=90, help='percentile for threshold')
def main(tmx_file, lang, percentile):
    """ Computes topic models by clustering dense word
        embeddings.
    """
    logger = logging.getLogger(__name__)
    logger.info('Compute topic model: {}, {}, {}'.format(
                tmx_file, lang, percentile))

    word_level_statistics = WordLevelStatistics(corpus_file=[text_fn[lang]],
                                                percentile_C=percentile)
    word_level_statistics.compute_spectra()

    lvls_df = pd.DataFrame(word_level_statistics.level_stat_thresholded)
    significant_terms = word_level_statistics.significant_terms
    print('Threshold: {}, ({} percentile) find {} significant terms.'.format(
                                 word_level_statistics.threshold,
                                 word_level_statistics.percentile_C,
                                 len(significant_terms)))

    vectors = {}
    for language in ['en']:
        vectors[language] = pymagnitude.Magnitude(language_model[language])

    significant_vectors = vectors[lang].query(significant_terms)

    try:
        fit = umap.UMAP(n_neighbors=15, n_components=target_dim,
                        metric='euclidean')
        data_d2v = fit.fit_transform(significant_vectors)
        fit = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean')
        vec_2d = fit.fit_transform(data_d2v)
    except Exception as ex:
        logging.error("Trying bhtsne. Got exception {}".format(ex))
        data_d2v = bhtsne.tsne(np.asfarray(significant_vectors,
                               dtype='float64'), dimensions=2)
        vec_2d = data_d2v

    lvls_df['vector'] = [v for v in data_d2v]

    significant_terms_enriched = enrich_significant_terms(
                                    lvls_df,
                                    data_d2v,
                                    vec_2d,
                                    cluster_selection_method)
    exemplar_scores, hovers = topic_exemplars(significant_terms_enriched)

    sents = [s['sentence'] for s in sentences]
    sent_ids = [s['sent_id'] for s in sentences]

    significant_terms_enriched['weight'] = significant_terms_enriched['sigma_nor']

    msg_topics = message_topics(topic_model=significant_terms_enriched,
                                sentences=sents,
                                sentences_ids=sent_ids,
                                significant_terms=significant_terms)

    msg_topics_df = pd.DataFrame(msg_topics).fillna(0.0).T

    K = significant_terms_enriched['topic'].max() + 1
    topics, top_columns = display_topics(significant_terms_enriched,
                                         n_rows=25, n_cols=K)

    pwd = os.environ.get('PWD')
    fmt = '{}/models/{}_{}_{}.csv'
    significant_terms_file_name = fmt.format(pwd, 'significant_terms',
                                             lang, str(percentile))
    msg_topics_file_name = fmt.format(pwd, 'msg_topics',
                                      lang, str(percentile))
    data_filename_fmt = '{}/models/significant_vectors_{}_{}.npy'
    data_filename = data_filename_fmt.format(pwd, lang, percentile)

    significant_terms_enriched.to_csv(significant_terms_file_name,
                                      index=False, encoding='utf-8')
    msg_topics_df.to_csv(msg_topics_file_name, index=False, encoding='utf-8')
    np.save(data_filename, data_d2v)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
