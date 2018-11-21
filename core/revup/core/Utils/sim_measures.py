"""
Collection of 
"""
import re
import math
from collections import Counter

import numpy as np

WORD = re.compile(r'\w+')

def sentences_intersection(sent1, sent2):
    """
   Calculate the intersection between 2 sentences

    """
    # split the sentence into words/tokens
    set1 = set(sent1.split(" "))
    set2 = set(sent2.split(" "))
    # If there is not intersection, just return 0
    if (math.log(len(set1)) + math.log(len(set2))) == 0:
        return 0

    # We normalize the result by the average number of words
    return len(set1.intersection(set2)) / (math.log(len(set1)) + math.log(len(set2)))


def sentence_cosine_similarity(sent1, sent2):
    """
    returns cosine similarity between two sentences
    """
    vec1 = text_to_vector(sent1)
    vec2 = text_to_vector(sent2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def cosine_similarity(u_vec, v_vec):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    denom = (math.sqrt(np.dot(u_vec, u_vec)) * math.sqrt(np.dot(v_vec, v_vec)))

    if not denom:
        return 0
    sim = np.dot(u_vec, v_vec) / denom
    return sim


def kl_divergence(p_dist, q_dist):
    """Returns Kullback-Leibler divergence D(P || Q) for discrete distributions
     p,q 
    """
    p_dist = np.asarray(p_dist, dtype=np.float)
    q_dist = np.asarray(q_dist, dtype=np.float)

    return np.sum(np.where(p_dist != 0, p_dist * np.log(p_dist / q_dist), 0))


def js_divergence(p_dist, q_dist):
    """Returns Jensen-Shannon divergence D(P || Q) for discrete distributions
     p,q 
    """
    p_dist = np.array(p_dist)
    q_dist = np.array(q_dist)
    distance1 = p_dist * np.log2(2 * p_dist / (p_dist + q_dist))
    distance2 = q_dist * np.log2(2 * q_dist/ (p_dist + q_dist))
    distance1[np.isnan(distance1)] = 0
    distance2[np.isnan(distance2)] = 0
    distance = 0.5 * np.sum(distance1 + distance2)
    return distance


def dice(word_a, word_b):
    """
    Returns dice coefficient between two words
    """
    len_a = len(word_a)
    len_b = len(word_b)
    a_chars = set(word_a)
    b_chars = set(word_b)
    overlap = len(a_chars & b_chars)
    return 2.0 * overlap / (len_a + len_b)


def text_to_vector(text):
    """
    Vectorizes text
    """
    words = WORD.findall(text)
    return Counter(words)
