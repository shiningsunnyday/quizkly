# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:27:12 2014

@author: giris_000
"""
import logging

from pattern.text.en import comparative, superlative, pluralize, singularize, conjugate, \
    PARTICIPLE, PRESENT, PAST, INFINITIVE, SINGULAR

LOGGER = logging.getLogger(__name__)

def convert(word, convert_tag):
    """ 
    Converts word of POS current_tag to be of POS convert_tag 
    e.g. NN to NNS, JJ to JJR
    """
    if convert_tag[0] == 'N':
        return convert_noun(word, convert_tag)
    elif convert_tag[0] == 'J':
        return convert_adj(word, convert_tag)
    elif convert_tag[0] == 'V':
        return convert_verb(word, convert_tag)
    else:
        LOGGER.debug("Method only works for nouns, adjectives and verbs")
        return word


def convert_noun(word, convert_tag):
    """ 
    Two possibilities: either singularize or pluralize 
    """
    vowel = ["a", "e", "i", "o", "u"]
    if convert_tag[-1] == 'S':
        if word[-1] == 's':
            return word
        return pluralize(word)
    elif convert_tag[-1] != 'S':
        if word[-1] != 's':
            return word
        if word[-1] == 's' and word[-2] in vowel:
            return word
        return singularize(word)
    else:
        return word


def convert_adj(word, convert_tag):
    """ 
    Two possibilities: convert to superlative; convert to comparative 
    """
    if convert_tag == 'JJR':
        return comparative(word)
    elif convert_tag == 'JJS':
        return superlative(word)
    else:
        return word


def convert_verb(word, convert_tag):
    """
    Convert verb to right tense
    """
    if convert_tag == 'VBD':
        return conjugate(word, PAST)
    elif convert_tag == 'VBG':
        pre_part = conjugate(word, PRESENT + PARTICIPLE)
        if word == pre_part:
            return conjugate(word, INFINITIVE)
        else:
            return pre_part
    elif convert_tag == 'VBN':
        return conjugate(word, PAST + PARTICIPLE)
    elif convert_tag == 'VBP':
        return conjugate(word, PRESENT, person=2, number=SINGULAR)
    elif convert_tag == 'VBZ':
        return conjugate(word, PRESENT, person=3, number=SINGULAR)
