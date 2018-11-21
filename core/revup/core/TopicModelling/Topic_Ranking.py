# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:47:01 2014

@author: Girish Kumar
"""

import heapq

def is_topical(topic_model, sentence):
    """ Checks if sentence has a peaked topic distribution """
    topic_dist = topic_model.get_topic_distribution(sentence)
    return is_dist_topical(topic_dist) and len(sentence.split(' ')) >= 3
    

def is_dist_topical(topic_dist):
    """Checks if a topic distribution is peaked"""
    max_three_topics = heapq.nlargest(3, topic_dist)
    # we store weighted sum of highest 3 probabilities
    topic_dist_max = \
        0.5 * max_three_topics[0] + \
        0.3 * max_three_topics[1] + \
        0.2 * max_three_topics[2]
    return topic_dist_max >= 0.4 

def get_topical_ones(topic_model, sentences):
    """ Returns only the topical sentences out of all the input sentences """
    topic_dists = topic_model.get_topic_distribution_batch(sentences)
    return [sent for i, sent in enumerate(sentences) if is_dist_topical(topic_dists[i])]


def topic_ranking(documents, topic_model):
    """ Returns sentences which are peaked from a list of sentences """
    return [sent for sent in documents if is_topical(sent, topic_model)]
