#CREATES A TOPIC LIST FROM A GROUP OF DOCUMENTS
import bz2, json, string
import sys, os, io, datetime, re
# import ijson
import sqlite3
from operator import itemgetter
from collections import Counter, defaultdict
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import pymagnitude
import spacy
import numpy
#to store UMAP Session Model
import joblib
model_file_name = 'sess_UMAPsav'
nlp = spacy.load('en')

from classes.nsaSrc.models import WordLevelStatistics

from classes.nsaSrc.models import (
            enrich_significant_terms,
            display_topics,
            topic_exemplars,
            hdbscan_parameter_search,
            enumerate_exemplars,
            topic_order_index,
)

# from sklearn.neighbors import NearestNeighbors

import hdbscan
try:
    import umap
    print("Using: umap")
except ImportError:
    import bhtsne

#LOGGING #####
import logging

import pdb

def topic_extractor(data_df,type_of_extraction):
    """Topic extractor function extracts topics from a csv file entered in file path.
    Depending on Type it can extract topics of individual papers or the whole session"""
    
    def topic_params_object(topics,words,vectors_300):
        #normalization parameters
        start_range = 1
        end_range = 10
        topic_parameters = []
        topics_ids = topics.columns.values.tolist()
        topic_weight = 0;
        for each_topic in topics:
            topic_average_vector = []
            this_topic = {
                'topic_id':each_topic,
                'vector':[],
                'vector300':[],
                'words':[],
                'weight':0
            }  
            this_topic['words'] = topics[each_topic].dropna().values.tolist()
            for word in this_topic['words']:
                if word[len(word)-1]=='*':
                    this_topic['vector'].append(words[words['word*']==word]['vector'].values.tolist()[0])
                    this_topic['weight'] = this_topic['weight'] + words[words['word*']==word]['sigma_nor'].values.tolist()[0]
                    this_topic['vector300'].append(words[words['word*']==word]['vector300'].values.tolist()[0])
                    # pdb.set_trace()
            this_topic['vector'] = numpy.mean(this_topic['vector'],axis=0)
            this_topic['vector300'] = numpy.mean(this_topic['vector300'],axis=0)
            topic_parameters.append(this_topic)
        #Normalize values between 0-1
        df_topic_parameters = pd.DataFrame(topic_parameters)
        df_topic_parameters_weight = df_topic_parameters['weight']
        df_topic_parameters['weight'] = (end_range-start_range)*(df_topic_parameters_weight-df_topic_parameters_weight.min())/(df_topic_parameters_weight.max()-df_topic_parameters_weight.min()) + start_range 
        # pdb.set_trace()
        return df_topic_parameters
    
    #SETUP LANGUAGE MODEL AND PIPELINE VARIABLES####
    lang = 'en'
    language_model = {'en':'./classes/nsaSrc/data/external/wiki-news-300d-1M.magnitude'}
    if type_of_extraction == 'session':
        percentile_C = 95
    else:
        percentile_C = 80
    target_dim = 10
    cluster_selection_method = 'leaf'

    #We only get the ones that have text
    data_df = data_df[data_df['text'] != 'Parsing Error']

    def en_filter(text):
        spacey_doc = nlp(text)
        # pdb.set_trace()
        sentences = []
        for sentence in spacey_doc.sents:
            for token in sentence:
                if not token.__len__() < 4 and not token.is_stop and not token.like_num and not token.is_digit:
                    sentences.append(str(token))
                # else:
                #     print(token)
        return sentences

    # data_subset = [record for record in data if record['Text'] != False and 'sagepub' in record['Text'].lower()]

    # data_subset = data_df
    text_fn = {'en':'./classes/nsaSrc/data/processed/en_flat.txt',
            'ko':'./classes/nsaSrc/data/processed/ko_flat.txt' }

    # THIS CODE CREATES THE DATA FOR PROCESSING AND STORES IT IN ./vizlit/data/processed AS A FLAT TEXT FILE ##########
    #
    #
    if type_of_extraction == 'session':
        with open(text_fn[lang], 'w', encoding='utf-8') as fp:
            for record_text in data_df['text']:
                # if record['Text'] != False:
                sentences = en_filter(record_text)
                for s in sentences:
                    fp.write(s + '\n')
    elif type_of_extraction == 'document':
        with open(text_fn[lang], 'w', encoding='utf-8') as fp:
            # for x in range(2):
            for record_text in data_df['text']:
                # if record['Text'] != False:
                sentences = en_filter(record_text) #needs to be an array to work
                for sent in sentences:
                    fp.write(sent + '\n')

    ##########################

    ####################################

    ##FIND SIGNIGICANT TERMS IN CORPUS #########
    word_level_statistics = WordLevelStatistics(corpus_file=[text_fn[lang]], percentile_C=percentile_C)
    word_level_statistics.compute_spectra()
    full_collection = pd.DataFrame(word_level_statistics.level_stat)
    lvls_df = pd.DataFrame(word_level_statistics.level_stat_thresholded)
    lvls_df['threshold']= word_level_statistics.threshold
    # pdb.set_trace()
    #Minimize corpus to most important words
    significant_terms = word_level_statistics.significant_terms

    #SOMETHING BROKE HERE FOR SOME REASON LVLS_DF IS ONE VALUE BIGGER THAN IT SHOULD AFTER FILTERING
    # if type_of_extraction == 'session':
    #Remove numbers and short words
    spacey_significant_terms = nlp(' '.join(word_level_statistics.significant_terms))
    significant_terms = significant_terms;
    significant_terms = []
    for sentence in spacey_significant_terms.sents:
        for token in sentence:
            if not token.__len__() < 4 and not token.is_stop and not token.like_num and not token.is_digit:
                significant_terms.append(str(token))
            # else:
            #     # pdb.set_trace()
            #     # print (token)
            #     #Remove token from dataframe
            #     # pdb.set_trace()
            #     lvls_df = lvls_df[lvls_df.word != str(token)]
    lvls_df_filtered = pd.DataFrame()
    for each_word in significant_terms:
        lvls_df_filtered = lvls_df_filtered.append(lvls_df[lvls_df.word == each_word])
    lvls_df = lvls_df_filtered
    # # print('With threshold = {}, ({} percentile) find {} significant terms.'.format(
    # #     word_level_statistics.threshold, word_level_statistics.percentile_C, len(significant_terms)))

    ##CLUSTER WORD EMBEDDINGS
    vectors = {}
    for l in ['en']:
        vectors[l] = pymagnitude.Magnitude(language_model[l])
    significant_vectors = vectors[lang].query(significant_terms)

    try:
        fit = umap.UMAP(n_neighbors=15, n_components=target_dim, metric='euclidean')
        data_d2v = fit.fit_transform(significant_vectors) #np.asfarray(significant_vectors, dtype='float64' ))
        if type_of_extraction == 'session':
            #store model
            joblib.dump(fit,model_file_name)
        fit = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean')
        vec_2d = fit.fit_transform(data_d2v)
    except Exception as ex:
        pdb.set_trace()
        logging.error("Trying with less dimensions. Got exception {}".format(ex))
        # data_d2v = bhtsne.tsne(np.asfarray(significant_vectors, dtype='float64' ),dimensions=2)
        # vec_2d = data_d2v
        #Try again with less neighbors this is just a TEMPORAL FIX
        fit = umap.UMAP(n_neighbors=7, n_components=target_dim, metric='euclidean')
        data_d2v = fit.fit_transform(significant_vectors)
        fit = umap.UMAP(n_neighbors=7, n_components=2, metric='euclidean')
        vec_2d = fit.fit_transform(data_d2v)
    try:
        lvls_df['vector'] = [v for v in data_d2v]
    except ValueError:
        pdb.set_trace()
        print('Error')
    lvls_df['vector300'] = [v for v in significant_vectors]
    significant_terms_enriched = enrich_significant_terms(lvls_df, data_d2v, vec_2d, cluster_selection_method)

    topics,top_columns = display_topics(significant_terms_enriched,n_rows=25,n_cols=250)
    # topics,top_columns = display_topics(significant_terms_enriched,n_rows=25,n_cols=10)#testing with 10 topics
    print('{} topics'.format(significant_terms_enriched['topic'].max() + 1))
    print('/n')
    print(topics)
    topic_params = topic_params_object(topics,lvls_df,significant_vectors)
    return {'topics':topics,'lvls_df':lvls_df,'topic_params':topic_params}