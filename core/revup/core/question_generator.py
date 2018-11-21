# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 20:43:39 2015

@author: giris_000
"""
import os
import re
import pickle
import logging

import gensim
from gensim.similarities.index import AnnoyIndexer

import kenlm
import spacy

from . import basepath
from .question import Question
from .GapSelection.GapSelectorModel import GapSelectorModel
from .GapSelection.sentence_features import SentenceFeatureComputer
from .DistractorSelection.distractor_features import DistractorFeatureComputer
from .TopicModelling import Topic_Ranking
from .TopicModelling.dlmodel import SAmodel
from .TopicModelling import Paragraph_Question as para_qn

LOGGER = logging.getLogger(__name__)

class QuestionGenerator(object):
    """
    Uses the full pipeline to generate questions 
    """
    def __init__(self, models_path=None):
        if models_path is None:
            models_path = os.path.join(basepath.get_base_path(), 'models')

        LOGGER.info("Loading DATM Model...")
        tm_path = os.path.join(models_path, 'dl30bio.pkl')
        self.topic_model = SAmodel.load(tm_path)

        LOGGER.info("Loading Word2Vec...")
        self.word_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
            os.path.join(models_path, 'wmdatabio70.bin'), binary=True)

        LOGGER.info("Loading Annoy Index...")
        self.ann_index = AnnoyIndexer()
        self.ann_index.load(os.path.join(models_path, 'wmdatabio70.ann'))
        self.ann_index.model = self.word_model

        LOGGER.info("Loading Forward Language Model...")
        self.lang_model = kenlm.Model(os.path.join(models_path, 'bio.bin'))

        LOGGER.info("Loading Reverse Language Model...")
        self.rev_lang_model = kenlm.Model(os.path.join(models_path, 'biorev.bin'))

        LOGGER.info("Loading Gap Classifier...")
        self.gap_selector = GapSelectorModel.load(os.path.join(models_path, 'biocontext.gsm'))

        LOGGER.info("Loading SpaCy Parser WITH GloVe Vectors...")
        self.parser = spacy.load('en')

        LOGGER.info("Loading Wiki Redirect Dict")
        with open(os.path.join(models_path, 'redirecttrie.pkl'), 'rb') as redirect_file:
            self.redirectdict = pickle.load(redirect_file, encoding='bytes') #hacky 2to3

    def get_question(self, sentence, previous_sentences=None, num_dists=4):
        """
        Given a sentence, generates a question with num_dists number of distractors
        """
        if not Topic_Ranking.is_topical(self.topic_model, sentence):
            return None
        
        proc_sent = self.clean_sentence(sentence)
        spacy_doc = self.parser(sentence)
        sfc = SentenceFeatureComputer(sentence, proc_sent, spacy_doc, word_model=self.word_model, 
                                      lang_model=self.lang_model, topic_model=self.topic_model)
        LOGGER.debug("Choosing Gaps...")
        LOGGER.debug("Loading Gap Selector Model...")
        gaps = self.gap_selector.select_gap(sfc, multi_gap_thres=0.05)
        sfc.best_gaps = gaps
        if gaps is None:
            return None

        LOGGER.debug("Getting Distractors...")
        dfc = DistractorFeatureComputer(sfc.doc, gaps, self.word_model, self.ann_index, self.parser, 
                                        self.redirectdict, None, None)
        final_gaps, distractors = dfc.filter_vocab(num_dists)
        if len(distractors) < num_dists:
            return None
        sfc.distractors = distractors

        LOGGER.debug("Question: %s", sfc.create_question(final_gaps))
        LOGGER.debug("Answer: %s", ' ... '.join([gap.text for gap in final_gaps]))
        LOGGER.debug("Options: %s", str(sfc.distractors))
        
        #if the sentence has a anaphora, pinpoint a context sentence
        
        context_sentence = ""
        if previous_sentences:
            context_sentence = para_qn.dep_context(previous_sentences, sfc.doc, self.parser)
            
        if len((context_sentence + " " + sfc.raw_text).split(' ')) > 40: 
            return None
        question = Question(sfc.create_question(final_gaps, context_sentence), 
                            ' ... '.join([gap.text for gap in final_gaps]), sfc.distractors)
        return question

    def get_questions_generator(self, sentences):
        """
        Returns literally a question generator
        """
        qnid = 0
        for i, sentence in enumerate(sentences):
            question = self.get_question(sentence, sentences[0:i])
            if question is not None:
                question.id = qnid
                qnid += 1
                yield question
    
    def get_questions(self, sentences):
        """
        Generates questions from multiple sentences
        """
        return [self.get_question(sentence, sentences[0:i]) for i, sentence in enumerate(sentences)
                if Topic_Ranking.is_topical(self.topic_model, sentence)]

    def get_question_from_doc(self, raw_text, proc_text, spacy_doc, num_dists=4, previous_sentences=None):
        """
        Generates a question form a spacy doc
        """
        sfc = SentenceFeatureComputer(raw_text, proc_text, spacy_doc, word_model=self.word_model, 
                                      lang_model=self.lang_model, topic_model=self.topic_model)
        LOGGER.debug("Choosing Gaps...")
        LOGGER.debug("Loading Gap Selector Model...")
        gaps = self.gap_selector.select_gap(sfc, multi_gap_thres=0.05)
        sfc.best_gaps = gaps
        if gaps is None:
            return None

        LOGGER.debug("Getting Distractors...")
        dfc = DistractorFeatureComputer(sfc.doc, gaps, self.word_model, self.ann_index, self.parser, 
                                        self.redirectdict, None, None)
        final_gaps, distractors = dfc.filter_vocab(num_dists)
        if len(distractors) < num_dists:
            return None
        sfc.distractors = distractors

        LOGGER.debug("Question: %s", sfc.create_question(final_gaps))
        LOGGER.debug("Answer: %s", ' ... '.join([gap.text for gap in final_gaps]))
        LOGGER.debug("Options: %s", str(sfc.distractors))
        
        #if the sentence has a anaphora, pinpoint a context sentence
        
        context_sentence = ""
        if previous_sentences:
            context_sentence = para_qn.dep_context(previous_sentences, sfc.doc, self.parser)
            
        if len((context_sentence + " " + sfc.raw_text).split(' ')) > 40: 
            return None
        question = Question(sfc.create_question(final_gaps, context_sentence), 
                            ' ... '.join([gap.text for gap in final_gaps]), sfc.distractors)
        return question

    def get_questions_batch_generator(self, sentences, batch_size=20, consider_context=True):
        """
        Yields batches of questions by processing subsets, of batch_size, of sentences
        """
        if len(sentences) <= batch_size:
            num_batches = 1
        else:
            num_batches = len(sentences)//batch_size
        for i in range(num_batches):
            batch = sentences[i*batch_size:(i+1)*batch_size]
            topical_batch = Topic_Ranking.get_topical_ones(self.topic_model, batch)
            proc_batch = [self.clean_sentence(sent) for sent in topical_batch]

            batch_questions = []
            for j, doc in enumerate(self.parser.pipe(topical_batch, batch_size=batch_size, 
                                    n_threads=4)):
                previous_sentences = None
                if consider_context:
                    previous_sentences = sentences[0: i*batch_size+j]
                batch_questions.append(self.get_question_from_doc(batch[j], proc_batch[j], doc,
                                       previous_sentences=previous_sentences))
            yield batch_questions

    @staticmethod
    def clean_sentence(sentence):
        """
        Cleans sentence for better spacy parser performance
        """
        # replacing brackets from text 
        # seems to interfere with parse tree brackets and simplifies sentence
        sentence = re.sub(r' \(.*?\) ', ' ', sentence)
        sentence = re.sub(r'\(.*?\)', '', sentence)
        if sentence[-1] == '.': #remove fullstop to improve parsing
            sentence = sentence[:-1]
        return sentence
