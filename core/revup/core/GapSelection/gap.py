# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 09:40:50 2014

@author: giris_000
"""


class Gap(object):
    def __init__(self, span, minimum_features = True):
        self.span = span
        self.text = self.span.orth_
        self.minimum_features = minimum_features
        self.rating = []
        self.corrected_rating = None
        self.rater = []


    def addRating(self, rating, rater):
        self.rating.append(int(rating))
        self.rater.append(rater)

    def vectorize(self):
        if self.minimum_features:
            v = [self.num_wn_sets, self.length]#, self.topic_kld_average]#self.conj]
            v += self.wordvec.tolist()
            v += self.childvec.tolist()
            #v += self.sentence_topic_dist.tolist()
            #v += self.topic_dist.tolist()
            return v

        v = [self.num_wn_sets, self.length, self.wlength, self.overlap, self.sentwords, self.word_overlap, self.index, self.topic_kld_average] #without index udoes better, word_overlap useless
        pos_tag_dict = {'CC': 0, 'CD': 1, 'DT': 2, 'NN': 3, 'NNP': 3, 'NNPS': 3, 'NNS': 3, 'FW': 4, 'IN': 5, 'JJ': 6,
                        'JJR': 6, 'JJS': 6,
                        'VBD': 7, 'VBG': 7, 'VBN': 7, 'VBP': 7, 'VBZ': 7, 'POS': 8, 'PRP': 9, 'PRP$': 9, 'RB': 10,
                        'RBR': 10, 'RBS': 10,
                        'WDT': 11, 'WP': 11, 'WP$': 11, 'WRB': 11}
        pos_vec = [0] * 12
        prev_pos_vec = [0] * 12
        post_pos_vec = [0] * 12

        for tag in self.tags:
            try:
                pos_vec[pos_tag_dict[tag]] = 1
            except:
                pos_vec[-1] = 0

        for tag in self.prevtag[0:2]:
            try:
                prev_pos_vec[pos_tag_dict[tag]] = 1
            except:
                prev_pos_vec[-1] = 0
        for tag in self.aftertag[0:2]:
            try:
                post_pos_vec[pos_tag_dict[tag]] = 1
            except:
                post_pos_vec[-1] = 0

        ner_dict = {'PERSON': 0, 'NORP': 1, 'FACILITY': 2, 'ORG': 3, 'GPE': 4, 'LOC': 5, 'EVENT': 6,
                    'WORK_OF_ART': 7, 'LANGUAGE': 8, 'DATE': 9, 'TIME': 9, 'PERCENT': 10, 'MONEY': 10, 
                    'QUANTITY': 10, 'CARDINAL': 10, 'ORDINAL': 11, 'O': 12}
        ner_vec = [0] * len(ner_dict)
        try:
            ner_vec[ner_dict[self.label]] = 1
        except:
            ner_vec[ner_dict['O']] = 1
            
        dep_dict = {'acl': 0, 'relcl': 0, 'acomp': 1, 'advcl': 2, 'advmod': 2, 'agent': 4, 'amod': 5, 'appos': 6, 
                   'attr': 7, 'aux': 30, 'auxpass': 30, 'case': 30, 'cc': 30, 'ccomp': 8, 'complm': 30, 
                   'compound': 9, 'conj': 10, 'csubj': 11, 'csubjpass': 11, 'dative': 12, 'dep': 30, 
                   'det': 30, 'dobj': 13, 'expl': 30, 'hmod': 14, 'hyph': 30, 'infmod': 30, 'intj': 30, 
                   'iobj': 15, 'mark': 30, 'meta': 30, 'neg': 30, 'nmod': 16, 'nn': 17, 'npadvmod': 18, 
                   'nsubj': 19, 'nsubjpass': 20, 'num': 21, 'number': 30, 'nummod': 22, 'oprd': 23, 
                   'parataxis': 30, 'partmod': 24, 'pcomp': 25, 'pobj': 26, 'poss': 27, 'possessive': 30, 
                   'preconj': 30, 'prep': 30, 'prt': 30, 'punct': 30, 'quantmod': 30, 'rcmod': 30, 
                   'root': 28, 'xcomp': 29, 'o': 30}
        dep_vec = [0] * (dep_dict['o']+1)
        for lbl in self.dep_labels:
            if lbl not in dep_dict:
                lbl = 'o'
                dep_vec[dep_dict[lbl]] = 1

        #v += prev_pos_vec + post_pos_vec #removing post_pos_vec increased performance
        #v += dep_vec + ner_vec #dep_vec & ner_vec seems useless  
        v +=self.childvec.tolist()
        v += self.sentence_topic_dist.tolist()
        v += self.topic_dist.tolist()
        v += self.wordvec.tolist()


        return v

    def __str__(self):
        return 'Text: %s Tags: %s' % (self.text, str(self.tags))
