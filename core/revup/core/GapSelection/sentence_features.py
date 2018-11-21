# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 19:20:35 2014

Sentence object to track important features
Features that were found to be not very powerful in the initial models have been commented out.
Final models trained on final data should confirm features

@author: giris_000
"""
import logging

import numpy
from nltk.corpus import wordnet as wn

from .gap import Gap
from ..Utils import preprocess as pr
from ..Utils.sim_measures import kl_divergence
from ..Utils.kenlm_helper import scoreword

LOGGER = logging.getLogger(__name__)

class SentenceFeatureComputer(object):
    """
    Filters words from sentence to obtain list of candidate gaps, 
    for which features are computed for the machine learning classifier.
    """
    def __init__(self, raw_text, proc_text, spacy_doc, lang_model=None, word_model=None, topic_model=None, minimum_features=True):
        self.raw_text = raw_text
        self.text = proc_text
        self.doc = spacy_doc
        self.minimum_features = minimum_features

        self.lang_model = lang_model
        self.word_model = word_model
        self.topic_model = topic_model

        self.gaps = []  # store potential gaps
        self.filter_gaps() # get a list of candidate gaps from the sentence

        # setting features for the gaps
        if self.minimum_features:
            self.set_min_features()
        else:
            self.set_syntactic_features()
            self.set_pos_features()
            self.set_ner_features()
            self.set_dep_features()
            if self.lang_model: 
                self.set_lm_features()
            self.set_wm_features()
            self.set_conj_features()
            self.set_tm_features()

        if len(self.gaps) > 0: 
            LOGGER.debug('Gap Vector Length: %d', len(self.gaps[0].vectorize()))


    def filter_gaps(self):
        """
        Filter gaps by whether cardinal, adjective, noun, verb. 
        Shouldn't be a stopword or duplicate in sentence
        """
        duplicate_idx = [0]*len(self.doc) #1 if chosen, 0 otherwise
        candidates = []

        #we first take noun phrases and named entities
        phrases = list(self.doc.ents) + list(self.doc.noun_chunks) #entities have highest precedence
        phrases = [self.backoff_phrase(phrase) for phrase in phrases]
        phrases = [phrase for phrase in phrases if phrase]
        LOGGER.info('Initial Phrase List: ' + '||'.join([phr.orth_ for phr in phrases]))

        for phrase in phrases:
            if phrase is None or len(phrase) == 0:
                continue

            if self.raw_text.count(phrase.orth_) > 2:
                LOGGER.debug('Deleted phrase because it occurs > twice in question sentence: ' 
                             + phrase.orth_)
                continue

            if any([t.tag_ == 'CD' for t in phrase]):
                LOGGER.debug('Deleted phrase because it has numbers. Temporary fix.')
                continue

            if 1 not in duplicate_idx[phrase.start:phrase.end]:
                candidates.append(phrase)
                duplicate_idx[phrase.start:phrase.end] = [1] * len(phrase)
            else:
                LOGGER.debug('Deleted phrase because duplicate: %s', phrase.orth_)


        #next look at tokens one by one
        for tok in self.doc:
            # no stopwords, duplicates and words without vectors are allowed
            if duplicate_idx[tok.i] == 1 or tok.is_stop:
                LOGGER.debug('Deleted token because duplicate/stopword: %s', tok.orth_)
                continue
            if self.word_model is not None and not self.has_wordvec(tok.orth_):
                LOGGER.debug('Deleted token because wordvec-less: %s', tok.orth_)
                continue
            if self.raw_text.count(tok.orth_) > 2:
                LOGGER.debug('Deleted token because it occurs > twice in question sentence: %s', 
                             tok.orth_)

            # only consider nouns, adjectives and foreign words 
            # (verbs no more cuz most verbs were pretty useless)
            tag = tok.tag_
            if tag and tag[0] not in ['N', 'J', 'F', 'R']: 
            #or tag == 'CD'): ---> temporarily remove number support
                LOGGER.debug('Deleted token because doesnt have valid POS: %s %s', tok.orth_, tag)
                continue

            #if fits above conditions, add it to cand. list
            candidates.append(self.doc[tok.i:tok.i+1]) #hack to convert token to span
        
        #look through cands and update gap list
        for phrase in candidates:
            tags = [t.tag_ for t in phrase]
            gap = Gap(phrase)
            gap.tags = tags
            self.gaps.append(gap)

        LOGGER.info('Chosen Gaps: ' + '||'.join([gap.__str__() for gap in self.gaps]))

    def backoff_phrase(self, phrase):
        """
        Given a spacy span-phrase, clean it up by removing determiners in front
        and get the longest subset that exists in the word_model
        """
        while len(phrase) > 0 and (phrase[0].is_stop and phrase[0].tag_[0] != 'N'): 
            #sometimes determiners end up in the front
            phrase = phrase[1:]

        if self.word_model is not None:
            LOGGER.debug('Before wordvec backoff: ' + phrase.orth_)
            phrase = self.backoff_by_wordvec(phrase)
            aftbackoff = phrase.orth_ if phrase is not None else 'None'
            LOGGER.debug('After wordvec backoff: ' + aftbackoff)

        return phrase

    def set_min_features(self):
        """ 
        Sets the bare minimum features which have proven to work well 
        i.e. wordvecs, number of wn synsets & index 
        """
        self.set_wm_features()
        self.set_conj_features()
        #self.set_tm_features()
        for gap in self.gaps:
            # - Length of Gap (number of characters)
            gap.length = len(gap.text)
            # - number of wn synsets
            gap.num_wn_sets = len(wn.synsets(gap.text.replace(' ', '_'), 
                                             pr.penn_to_wn(gap.tags[-1])))


    def set_syntactic_features(self):
        """
        For each filtered gap, add length features
        """
        for gap in self.gaps:
            # - Length of Gap (number of characters)
            gap.length = len(gap.text)
            # - Word Length of Gap
            gap.wlength = len(gap.text.split(' '))
            # - gap character overlap with sentnece
            gap.overlap = float(gap.length) / len(self.text)
            # - number of words in sentence gap belongs to
            gap.sentwords = len(self.text.split(' '))
            # - word overlap
            gap.word_overlap = float(len(gap.span)) / gap.sentwords
            # - numerical position of first word in sentence
            gap.index = gap.span.start
            # - number of wn synsets
            num_wn_sets = len(wn.synsets(gap.text.replace(' ', '_'), pr.penn_to_wn(gap.tags[-1])))
            if num_wn_sets == 0:
                num_wn_sets = min([len(wn.synsets(w, pr.penn_to_wn(gap.tags[i]))) 
                                   for i, w in enumerate(gap.text.split(' '))])
            gap.num_wn_sets = num_wn_sets

    def set_pos_features(self):
        """
        For each filtered gap, add pos tag features
        """
        for gap in self.gaps:
            gap.prevtag = []
            gap.aftertag = []
            for idx in range(gap.span.start - 2, gap.span.start):
                if idx < 0:
                    gap.prevtag.append('O')
                    continue
                gap.prevtag.append(self.doc[idx].tag_)

            for idx in range(gap.span.end, gap.span.end + 2):
                if idx >= len(self.doc):
                    gap.aftertag += ['O']*(2-len(gap.aftertag))
                    break
                gap.aftertag.append(self.doc[idx].tag_)


    def set_ner_features(self):
        """
        For each filtered gap, compute and set NER tags
        """
        for gap in self.gaps:
            if gap.span[0].ent_type == 0: #spacy tags all relevant tokens
                tag = 'O' # no tag
            else:
                tag = gap.span[0].ent_type_
            gap.label = tag


    def set_lm_features(self):
        """
        For each filtered gap, compute and set the language model features
        """
        for gap in self.gaps:
            # - langmodel prob
            history = self.text.split(' ')[0:gap.index]
            if len(history) > 1:
                history = tuple(history)
            else:
                history = None
            words = str(gap.span.orth_).split()
            try:
                gap.transitionProb = scoreword(self.lang_model, tuple(words), history)
            except KeyError:
                gap.transitionprob = 0

    def backoff_by_wordvec(self, span, sliding_window=True):
        """ 
        For a phrase, finds the longest subphrase (of len > 1) that has a word vector. 
        None is returned if it's not found 
        """
        if not sliding_window:
            return self.non_sliding_backoff_by_wordvec(span)
        else:
            window_size = 4
            longest_span = None
            while not longest_span and window_size > 1:
                longest_span = self.sliding_backoff_by_wordvec(span, window_size)
                window_size -= 1
            return longest_span


    def non_sliding_backoff_by_wordvec(self, span):
        """
        For a phrase, remove words from front or back to find longest string
        with word vector. 
        None is returned if its not found.
        """
        if len(span) == 1 and self.has_wordvec(span.orth_):
            return span
        
        forward_span, backward_span = span, span
        while len(forward_span) > 1:
            if self.has_wordvec(forward_span.orth_):
                break
            forward_span = forward_span[1:]

        while len(backward_span) > 1:
            if self.has_wordvec(backward_span.orth_):
                break
            backward_span = backward_span[:-1]

        if len(backward_span) == 1 and len(forward_span) == 1:
            return None

        if len(backward_span) >= len(forward_span):
            return backward_span
        else:
            return forward_span

    def sliding_backoff_by_wordvec(self, span, window_size):
        """
        For a phrase, find longest window of window_size that has a word vector
        None is return if its not found
        """  
        if self.has_wordvec(span.orth_):
            return span

        if len(span) <= window_size:
            if self.has_wordvec(span.orth_):
                return span
            else:
                return None  
        # forward pass
        startidx = 0
        while startidx + window_size <= len(span):
            if self.has_wordvec(span[startidx:startidx+window_size].orth_):
                return span[startidx:startidx+window_size]
            startidx += 1    
        # backward pass
        startidx = len(span) - window_size
        while startidx >= 0:
            if self.has_wordvec(span[startidx:startidx+window_size].orth_):
                return span[startidx:startidx+window_size]
            startidx -= 1
        return None

    def has_wordvec(self, text):
        """ 
        For the given text, check if there's an entry in the word vector model
        """
        return text.replace(' ', '_') in self.word_model


    def set_wm_features(self):
        """
        For each filtered gap, compute and set the wordvec and wordvecs of context words
        """
        for gap in self.gaps:
            # set vector of gap
            gap.wordvec = self.get_word_vector(gap.span)
            # set parent vector of gap
            gap.parentvec = self.get_word_vector(gap.span[-1].head)
            # set average of child vectors
            childvec = numpy.asarray([0.0] * len(gap.wordvec))
            numchildren = 0
            for child in gap.span[-1].children:
                childvec += self.get_word_vector(child)
                numchildren += 1
            numchildren = 0
            for child in gap.span[-1].children:
                childvec += self.get_word_vector(child)
                numchildren += 1
            gap.childvec = childvec/numchildren if numchildren > 0 else childvec


    def set_tm_features(self):
        """
        Sets topic model related features
        """
        sentence_topic_dist = self.topic_model.get_topic_distribution(self.text)
        for gap in self.gaps:
            gap.sentence_topic_dist = sentence_topic_dist
            gap.topic_dist = self.topic_model.get_topic_distribution(gap.text)
            topic_kld_left = kl_divergence(sentence_topic_dist, gap.topic_dist)
            topic_kld_right = kl_divergence(gap.topic_dist, sentence_topic_dist)
            gap.topic_kld_average = (topic_kld_right + topic_kld_left)/2.0 


    def get_word_vector(self, span):
        """
        Given a spacy span, returns a vector from word2vec model, if provided,
        else a vector from spacy's glove model
        """
        if self.word_model:
            # word2vec neural embedding of gap
            try:
                return self.word_model[span.orth_.replace(' ', '_')]
            except KeyError:
                return numpy.asarray([0.0] * 70)
        else:
            # spacy's glove vecs
            return span.vector


    def set_dep_features(self):
        """
        For each filtered gap, compute and set its dependency label
        """       
        for gap in self.gaps:
            gap.dep_labels = [t.dep_.lower() for t in gap.span]
            #setting distance from root
            span_highest = gap.span.root
            root_dist = 0
            while span_highest.head is not span_highest:
                root_dist += 1
                span_highest = span_highest.head
            gap.root_dist = root_dist


    def set_conj_features(self):
        """
        For each filtered gap, set if it/or any of its children is a conj
        """
        for gap in self.gaps:
            gap.conj = 0
            if gap.span[-1].dep_ == 'conj':
                gap.conj = 1
            children = list(gap.span[-1].children)
            for child in children:
                if child.dep_ == 'conj':
                    gap.conj = 1


    def create_question(self, gaps, context_sentence=""):
        """
        Returns a fill-in-the blank question from the sentence and the gap passed into the method.
        """
        question = (context_sentence + " " + self.text)
        for gap in gaps: 
            question = question.replace(gap.text, '_'*8)
        return question.strip()
