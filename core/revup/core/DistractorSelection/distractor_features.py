# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 19:39:25 2014

@author: giris_000
"""
import re
import string
import itertools
import logging
from operator import itemgetter

import numpy as np
from nltk.corpus import wordnet as wn

from ..Utils import preprocess as pr, sim_measures as sm
from ..Utils.kenlm_helper import scoreword

LOGGER = logging.getLogger(__name__)
PUNC_REGEX = re.compile('[%s]' % re.escape(string.punctuation))

class DistractorFeatureComputer(object):
    """
    Class that deals with distractor selection for chosen gaps
    """
    def __init__(self, spacy_doc, gaps, word_model, ann_index, parser, redirect_trie, lang_model = None, rev_lang_model = None):
        self.word_model = word_model
        self.ann_index = ann_index
        self.lang_model = lang_model
        self.rev_lang_model = rev_lang_model
        self.redirect_trie = redirect_trie
        self.parser = parser
        self.distractors = []
        self.doc = spacy_doc
        self.sentence = pr.remove_punct(self.doc.text)
        self.gaps = gaps
        self.top_candidates = [] #stores top candidates for each gap in gaps

    def filter_vocab(self, num_dists=4):
        """
        Retrieves and compiles k distractors for the provided gaps where k is num_dists
        Returns the gaps for which distractors were generated for as well
        """
        distractor_sets = []
        chosen_gaps = []
        for gap in self.gaps:
            g_distractors = self.filter_vocab_gap(gap, num_dists)
            if len(g_distractors) < num_dists:
                continue
            distractor_sets.append(g_distractors)
            chosen_gaps.append(gap)
        chosen_gap_texts = [gap.text for gap in chosen_gaps]
        if len(distractor_sets) < 1:
            return [], []
        self.distractors = self.compile_distractor_sets(chosen_gap_texts, distractor_sets, 
                                                        num_dists)
        LOGGER.info('Chosen Distractors')
        LOGGER.info(self.distractors)
        return chosen_gaps, self.distractors

    @staticmethod
    def compile_distractor_sets(gap_texts, distractor_sets, num_dists=4):
        """
        Given distractors for one or two gaps, combine them
        """
        if len(distractor_sets) == 1:
            return distractor_sets[0]
        distractors = gap_texts + [""]*(num_dists-2)
        distractors[0] = distractors[0] + ' ... ' + distractor_sets[1].pop(0)
        distractors[1] = distractor_sets[0].pop(0) + ' ... ' + distractors[1]
        for i in range(2, num_dists):
            distractors[i] = distractor_sets[0].pop(0) + ' ... ' + distractor_sets[1].pop(0)
        return distractors 

    def filter_vocab_gap(self, gap, num_dists=4):
        """ 
        Using word_model to get an n-best list where n = 40. 
        Then, we filter by certain properties including contextual fit and lexical similarity 
        """
        n_best_list = []
        try:
            phrase = gap.text.replace(' ', '_')
            
            #retrieving candidate distractors from wordvector model
            if phrase[0].isupper() and phrase[1].islower():
                phrase_low = phrase.lower()
                try:
                    #n_best_list = self.word_model.most_similar_cosmul(positive=[phrase_low], 
                                                                      #topn=20)
                    query = self.word_model[phrase_low]
                    n_best_list = self.word_model.most_similar([query], topn=20, 
                                                               indexer=self.ann_index)
                except KeyError:
                    n_best_list = []

            if not n_best_list:
                query = self.word_model[phrase]
                n_best_list = self.word_model.most_similar([query], topn=20, 
                                                           indexer=self.ann_index)
                #n_best_list = self.word_model.most_similar_cosmul(positive=[phrase], topn=20)

        except KeyError:
            LOGGER.warning("Gap not in WordModel! No Distractors for this question!")
            return []

        # remove punctuation from distractors lest they exist
        # underscore connects constituent words in phrases so replace that with a space
        candidates = [[pr.remove_punct(pair[0].replace('_', ' ')), pair[1]] 
                      for i, pair in enumerate(n_best_list)]
        LOGGER.debug("Initial List")
        LOGGER.debug(candidates)
        # remove empty candidates that might have just been punctuations
        candidates = [[c[0], c[1]] for c in candidates if c[0].strip() != '']
        # removing words that might already exist in the sentence
        candidates = self.filter_words_in_sent(candidates)
        LOGGER.debug("Words in Sent")
        LOGGER.debug(candidates)

        # filtering by postag
        candidates = self.filter_pos(gap, candidates)
        LOGGER.debug("POS")
        LOGGER.debug(candidates)
        
        # filtering distractors with stopwords
        candidates = self.filter_stopword(candidates)
        LOGGER.debug("StopWords")
        LOGGER.debug(candidates)
        
        # preparing wordnet synsets for subsequent filters
        candidates, gap_syn, gap_hypomeronyms = self.prep_wordnet_synsets(gap, candidates)

        # filtering by wordnet similarity
        candidates = self.filter_wordnetsim(candidates, gap_syn)
        LOGGER.debug("WordnetSim")
        LOGGER.debug(candidates)

        # filtering meronyms and hyponyms
        candidates = self.filter_hypomeronyms(candidates, gap, gap_syn, gap_hypomeronyms)
        LOGGER.debug("hyponyms")
        LOGGER.debug(candidates)

        # rescoring
        candidates = self.rescore(candidates, gap)

        # filtering duplicates of both gaps and other distractors
        while len(candidates) >= num_dists: 
            #only check first num_dists+2 distractors for dupes. if there isnt any, loop terminates
            reserve_distractors = candidates[num_dists+2:] 
            candidates = self.filter_duplicates(candidates[0:num_dists+2], gap, gap_syn) 
            num_reserve = max(num_dists - len(candidates), 0)
            if num_reserve == 0: 
                break
            candidates = candidates + reserve_distractors[0:num_reserve]
        
        self.top_candidates.append(candidates) #keep for performance purposes
        # take strings of top num_dists distractors
        final_distractors = [d[0] for d in candidates[:num_dists]]

        # post process gaps
        final_distractors = self.post_process(final_distractors, gap)

        LOGGER.debug("Chosen Distractors for Gap %s", gap.text)
        LOGGER.debug(final_distractors)

        return final_distractors

    def rescore(self, candidates, gap):
        """
        Rescoring by context score and lexical similarity
        """
        context_scores = self.get_context_scores(gap, candidates)
        lex_sim = self.get_lex_sims(gap, candidates)

        for i, _ in enumerate(candidates):
            candidates[i][1] = (2.5/6 * candidates[i][1] + 
                                5.0/6 * context_scores[i] + 
                                1.5/6 * lex_sim[i])

        candidates = sorted(candidates, key=itemgetter(1), reverse=True) 
        return candidates

    def post_process(self, distractors, gap):
        """
        Deals with the capitalising, pos-transformation of gaps
        """
        prev_token, after_token = None, None
        if gap.span.start > 1:
            prev_token = self.doc[gap.span.start-1]
        if gap.span.end < len(self.doc):
            after_token = self.doc[gap.span.end]

        for i, distractor in enumerate(distractors):
            # ensure capitalization same as gap-phrase
            if gap.text[0].isupper():
                distractors[i] = distractor.capitalize()
            # ensure none of 
            distractor_words = distractor.split(' ')
            if (prev_token and 
                    pr.stem_doc(distractor_words[0:1]) == pr.stem_doc([prev_token.orth_])):
                distractors[i] = ' '.join(distractor_words[1:])
            if (after_token and 
                    pr.stem_doc(distractor_words[-1:]) == pr.stem_doc([after_token.orth_])):
                distractors[i] = ' '.join(distractor_words[:-1])
        return distractors

    def filter_words_in_sent(self, distractors):
        """
        Removes distractor candidates that appear in sentence 
        """
        stemmed_sentence = pr.stem_doc(self.sentence.split())
        filtered_distractors = []

        for pair in distractors:
            phrase = pair[0]
            phrase = phrase.split(' ')
            stemmed_phrase = pr.stem_doc(phrase)
            if not all(w in stemmed_sentence for w in stemmed_phrase):
                filtered_distractors.append(pair)

        return filtered_distractors

    def filter_pos(self, gap, distractors):
        """
        Removes distractor candidates that have different 
        part of speech from selected gap
        """

        ref_tag = gap.tags

        filtered_distractors = []
        phrases = [pair[0] for pair in distractors]
        tokenized = self.parser.tokenizer.pipe(phrases, batch_size=len(phrases), 
                                               n_threads=4)
        tagged_phrases = self.parser.tagger.pipe(tokenized, batch_size=len(phrases), 
                                                 n_threads=4)
        for pair in distractors:
            tagged = next(tagged_phrases)
            tags = [token.tag_ for token in tagged]   
            if 'CD' in tags: #ignore numbers for now
                continue
            # we check the first letter of the tag cos we dont care about plurality and so on
            if ref_tag[-1][0] == tags[-1][0]: # or (tags[0][0] == 'N' and ref_tag[0] == tags[0][0]):
                filtered_distractors.append(pair)
        return filtered_distractors

    @staticmethod
    def filter_stopword(distractors):
        """
        Remove stopwords from distractors and if distractor only contains stopword,
        remove the distractor
        """
        stoplist = pr.get_stop_words()
        filtered_distractors = []
        for pair in distractors:
            phrase = pair[0].split(' ')
            phrase = [w for w in phrase if w not in stoplist]
            if len(phrase) > 0:
                filtered_distractors.append(pair)
        return filtered_distractors

    def filter_duplicates(self, distractors, gap, gap_syn):
        """
        Remove distractors which are duplicates of gaps/other distractors
        """
        filtered_distractors = []
        delete_idx = []
        for i, dist in enumerate(distractors):
            if i in delete_idx:
                continue

            if self.check_duplication([dist[0], dist[2]], [gap.text, gap_syn]):
                delete_idx.append(i)
                continue
            
            for j, dist_check in enumerate(distractors):
                if i == j:
                    continue
                if self.check_duplication([dist[0], dist[2]], [dist_check[0], dist_check[2]]):
                    # delete the one with the lower wordvec score
                    if dist[1] > dist_check[1]:
                        delete_idx.append(j)
                    else:
                        delete_idx.append(i)
                    break

        filtered_distractors = [dist for i, dist in enumerate(distractors) if i not in delete_idx]

        return filtered_distractors

    def filter_hypomeronyms(self, distractors, gap, gap_syn, gap_hypomeronyms):
        """ 
        Removes distractors that are hyponyms of the gap to avoid confusion 
        """
        filtered_distractors = []

        for dist in distractors:
            dist_str, dist_syn = dist[0], dist[2]
            gap_str, gap_syn = gap.text, gap_syn
            
            if (len(gap_str.split(' ')) > 1 and len(dist_str.split(' ')) > 1 and 
                    pr.stem_doc(gap_str.split(' ')[-1:]) == pr.stem_doc(dist_str.split(' ')[-1:])):
                dist_str = ' '.join(dist_str.split(' ')[:-1]) 
                gap_str = ' '.join(gap_str.split(' ')[:-1])
                ref_tag = gap.tags[-2]
                dist_syn = wn.synsets(dist_str.replace(" ", "_"), pr.penn_to_wn(ref_tag))
                gap_syn = wn.synsets(gap_str.replace(" ", "_"), pr.penn_to_wn(ref_tag))
                gap_hypomeronyms_first = []
                for syn in gap_syn: 
                    gap_hypomeronyms_first += self.get_hypomeronyms(syn)
                if self.check_hypomeronym([gap_str, gap_syn], [dist_str, dist_syn], 
                                          gap_hypomeronyms_first):
                    continue

            if not self.check_hypomeronym([gap_str, gap_syn], [dist_str, dist_syn], 
                                          gap_hypomeronyms):
                filtered_distractors.append(dist)

        return filtered_distractors

    def filter_wordnetsim(self, distractors, gap_syn):
        """ 
        Removes distractors which are too far away from the gap in WordNet 
        """
        filtered_distractors = []
        for dist in distractors:
            if self.wordnet_sim(dist[2], gap_syn) >= 0.1:
                filtered_distractors.append(dist)
        return filtered_distractors            
    
    @staticmethod
    def prep_wordnet_synsets(gap, distractors):
        """
        Queries wordnet synsets for gap and distractors
        """
        ref_tag = gap.tags[-1]
        gap_syn = wn.synsets(gap.text.replace(' ', '_'), pr.penn_to_wn(ref_tag))
        gap_hypomeronyms = []
        for syn in gap_syn:
            gap_hypomeronyms += DistractorFeatureComputer.get_hypomeronyms(syn)
        for dist in distractors:
            dist.append(wn.synsets(dist[0].replace(" ", "_"), pr.penn_to_wn(ref_tag)))
        return distractors, gap_syn, gap_hypomeronyms
    
    @staticmethod
    def get_hypomeronyms(syn):
        """
        Return list of mero, hypo, holo, similar_tos for a certain synset 
        """
        hypomeronyms = []
        hypomeronyms += [i for i in syn.closure(lambda s: s.hyponyms())]
        hypomeronyms += [i for i in syn.closure(lambda s: s.part_meronyms())]
        hypomeronyms += [i for i in syn.closure(lambda s: s.member_holonyms())]
        hypomeronyms += syn.similar_tos()
        return hypomeronyms

    @staticmethod
    def wordnet_sim(set_a, set_b):
        """ 
        Computes maximum similarity between two synsets (s,t) spanning multiple 
        senses of two different words 
        """
        # permutate all possible sim calcs
        #possible_pairs = map(None, itertools.product(set_a, set_b)) 
        possible_pairs = itertools.product(set_a, set_b)
        scores = []
        for pair in possible_pairs:
            score = pair[0].path_similarity(pair[1])
            if score is not None:
                scores.append(score)
        if scores:
            return max(scores)
        else:
            return 0.1
    
    def check_duplication(self, word_x, word_y):
        """
        Checks if words are duplicates of one another by stemming and through WordNet
        """
        x_str, x_sn, y_str, y_sn = word_x[0], word_x[1], word_y[0], word_y[1]
        x_str = PUNC_REGEX.sub(' ', x_str)
        y_str = PUNC_REGEX.sub(' ', y_str)
        same_word = (''.join(pr.stem_doc(x_str.lower().split(' '))) == 
                     ''.join(pr.stem_doc(y_str.lower().split(' '))))

        if x_sn is not None and y_sn is not None: #only compare if word has a synset in wordnet
            same_synset = not set(x_sn).isdisjoint(set(y_sn)) 
        else:
            same_synset = False
        
        #check if either x or y redirects to the other in wikipedia
        redirect_queries = (x_str.lower().replace(' ', '_') + ';' + y_str.lower().replace(' ', '_'),
                            y_str.lower().replace(' ', '_') + ';' + x_str.lower().replace(' ', '_'))

        same_wiki = any([self.redirect_trie.get(query) for query in redirect_queries])
        return same_word or same_synset or same_wiki
    
    @staticmethod
    def check_hypomeronym(gap, dist, gap_hypomeronyms=None):
        """ 
        Check if the dist is a hyponym or meronym of gap 
        """
        g_str, g_sn, d_str, d_sn = gap[0], gap[1], dist[0], dist[1]
        g_str = PUNC_REGEX.sub(' ', g_str)
        d_str = PUNC_REGEX.sub(' ', d_str)
        if all([w in d_str.lower() for w in g_str.lower().split(' ')]):
            return True
            
        d_str_stemmed = pr.stem_doc(d_str.lower().split(' '))
        g_str_stemmed = pr.stem_doc(g_str.lower().split(' '))
        if all([w in d_str_stemmed for w in g_str_stemmed]):
            return True

        if not(g_sn != None and d_sn != None):
            return False

        if gap_hypomeronyms is None:
            gap_hypomeronyms = []
            for syn in g_sn: 
                gap_hypomeronyms += DistractorFeatureComputer.get_hypomeronyms(syn)

        if not set(d_sn).isdisjoint(set(gap_hypomeronyms)):
            return True

        return False

    def get_context_scores(self, gap, candidates):
        """ 
        Uses the language model (if provided) or the word vector model (if no lm)
        to measure how well the given distractor fits in the question sentence 
        """

        if self.lang_model: 
            LOGGER.debug("Lang Model provided. Transition probabilities used for contextual fit")
            return self.get_context_scores_lm(gap, candidates)
        else:
            LOGGER.debug("Lang Model NOT provided. WordVec similarity used for contextual fit")
            return self.get_context_scores_glove(gap, candidates)
           
        return []

    def get_context_scores_wm(self, gap, candidates):
        """ 
        Computes wordvec similarity with between the word vector of the candidate and 
        other words in the sentence 
        """
        gap_phrase = gap.text.replace(' ', '_')
        
        context_scores = []
        for candidate in candidates:
            context_score = 0.0
            for token in self.doc:
                if token.i < gap.span.end and token.i >= gap.span.start: 
                    continue
                if token.is_stop or token.is_punct: 
                    continue
                try:
                    sim = self.word_model.similarity(gap_phrase, candidate[0].replace(" ", "_"))
                    positional_weight = 0
                    if token.i < gap.span.start:
                        positional_weight = 1.0/(gap.span.start - token.i)
                    else:
                        positional_weight = 1.0/(token.i - gap.span.end + 1)
                    context_score += positional_weight * sim
                except KeyError:
                    context_score += 0
            context_scores.append(context_score)

        return context_scores

    def get_context_scores_glove(self, gap, candidates):
        """ 
        Computes each candidate's co-occurence scores with every other word in the sentence by 
        the dot product between their glove vectors 
        """        
        context_scores = []
        for candidate in candidates:
            context_score = 0.0
            cand_span = self.parser(candidate[0])
            for token in self.doc:
                if token.i < gap.span.end and token.i >= gap.span.start: 
                    continue
                if token.is_stop or token.is_punct: 
                    continue
                try:
                    print(token.vector.shape)
                    print(cand_span[0].vector.shape)
                    cooccurence = sum([np.dot(cand_token.vector, token.vector)/50.0 
                                       for cand_token in cand_span])
                    # average over length of span
                    cooccurence = cooccurence/len(cand_span)
                    positional_weight = 0
                    if token.i < gap.span.start:
                        positional_weight = 1.0/(gap.span.start - token.i)
                    else:
                        positional_weight = 1.0/(token.i - gap.span.end + 1)
                    context_score += positional_weight * cooccurence
                except KeyError:
                    context_score += 0
            context_scores.append(context_score)
        return context_scores

    def get_context_scores_lm(self, gap, candidates):
        """ 
        Returns n-gram transition probability of word given words that appear 
        before and words that appear after (revlm) 
        """
        history = str(self.sentence).split(' ')[0:gap.span.start]
        if len(history) >= 1:
            history = tuple(history)
        else:
            history = None

        rev_history = str(self.sentence).split(' ')[gap.span.start+1:]
        if len(rev_history) >= 1:
            rev_history = tuple(rev_history)
        else:
            rev_history = None

        context_scores = []

        transitionprob = 0
        for candidate in candidates:
            words = str(candidate[0]).split()
            if history:
                abs_score = abs(scoreword(self.lang_model, tuple(words), history))
                transitionprob = (15.0 - abs_score)/15
            else:
                transitionprob = 0
            if rev_history and self.rev_lang_model:
                abs_score = abs(scoreword(self.rev_lang_model, tuple(words), rev_history))
                transitionprob += (15.0 - abs_score)/15
            else:
                transitionprob = transitionprob
            context_scores.append(transitionprob)

        return context_scores
    
    @staticmethod
    def get_lex_sims(gap, candidates):
        """
        Uses the dice coefficient to measure lexical similarity between the gap and each distractor
        """
        return [sm.dice(c[0], gap.text) for c in candidates]
