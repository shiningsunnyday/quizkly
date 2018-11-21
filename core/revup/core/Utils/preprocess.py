# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:40:52 2014

@author: giris_000
"""

import os

import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk import stem
from nltk.stem import WordNetLemmatizer

from .. import basepath

with open(os.path.join(basepath.get_base_path(), 'data/stopwords.txt')) as f:
    STOPLIST = f.read().split(',')
STEMMER = stem.snowball.SnowballStemmer("english")

def lemmatize(document):
    """
    Lemmatizes input sentence
    """
    # do postaggin on input doc
    pos_result = nltk.pos_tag(document)
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    document = [lemmatizer.lemmatize(p[0], penn_to_wn(p[1])) for p in pos_result]
    return document


def stem_doc(document):
    """
    Stems input sentence
    """
    document = [STEMMER.stem(word.lower()) for word in document]
    return document


def remove_punct(sentence):
    """ 
    Removes punctuation from sentence 
    """
    for punct in [',', '.', ':', '(', ')', ';', '?', '-', '"']:
        sentence = sentence.replace(punct, ' ').strip()
    return sentence


def process_sentences(documents):
    """
    Removes unnecessary sentences and words from a list of sentences
    """
    # remove sentences lesser than 3 words
    documents = [document for document in documents if len(document) > 3]
    # convert to lowercase
    documents = [document.lower() for document in documents]
    # remove stopwords
    documents = [[word for word in word_tokenize(document) if word not in STOPLIST]
                 for document in documents]
    # remove punctuations
    documents = [[word for word in document if word.isalpha()]
                 for document in documents]
    # lemmatize sentences
    documents = [stem_doc(document) for document in documents]
    return documents


def clean_sentence(document):
    """
    Lowercases, stems, removes stopwords/punctuation from sentence 
    """
    # convert to lowercase
    document = document.lower()
    # remove stopword
    document = [word for word in word_tokenize(document) if word not in STOPLIST]
    # remove punctuation
    document = [word for word in document if word.isalpha()]
    # stem words
    return " ".join(stem_doc(document))


def get_stop_words():
    """
    Returns stop word list
    """
    return STOPLIST


def is_noun(tag):
    """
    Checks if tag is noun
    """
    return len(tag) > 0 and tag[0] == 'N'


def is_verb(tag):
    """
    Checks if tag is verb
    """
    return len(tag) > 0 and tag[0] == 'V'


def is_adverb(tag):
    """
    Checks if tag is adverb
    """
    return len(tag) > 0 and tag[0] == 'R'


def is_adjective(tag):
    """
    Checks if tag is adjective
    """
    return len(tag) > 0 and tag[0] == 'J'


def penn_to_wn(tag):
    """
    Converts tag to wordnet tag for lemmatization
    """
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN
