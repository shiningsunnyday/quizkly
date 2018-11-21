# import Autoencoder
import operator
import pickle
import itertools
import math
import numpy

import logging
logger = logging.getLogger(__name__)

from sklearn.feature_extraction.text import CountVectorizer

from .DBN import train_DBN

from ..Utils import preprocess as pr


class SAmodel(object):
    #######################################################################################
    """ Initialize and train the Autoencoder topic model with the above parameters """

    def __init__(self, hidden, max_iterations, num_topics, documents=None, file_path=None, input_data=None,
                 architecture='dbn-auto', sparsity=0.03, selectivity=0.03, df=2, max_f=2000):

        self.vectorizer = CountVectorizer(min_df=df, max_features=max_f, binary=True)
        if file_path:
            with open(file_path) as f:
                documents = f.read().splitlines()
            self.raw_documents = [document for document in documents if len(document.split()) > 3]
            documents = pr.process_sentences(documents)
            self.documents = [" ".join(document) for document in documents]
            self.features = self.vectorizer.fit_transform(self.documents)
        else:
            self.documents = documents
            self.features = self.vectorizer.fit_transform(self.documents)
        self.df = df
        self.max_iterations = max_iterations
        self.sparsity = sparsity
        self.num_topics = num_topics
        self.architecture = architecture
        self.hidden = hidden
        self.encoders = []
        self.dbn = 0
        logger.debug(self.getName())
        if self.architecture == 'stacked-auto':
            logger.error("Autoencoder not installed")
            return  # TODO: unreachable autoencoder
            # print 'stacked-auto'
            # i = 0
            # if not input_data:
            #     input_data = self.features.toarray()[1:100,:]
            # numpy.random.shuffle(input_data)
            # activation_function = T.nnet.sigmoid
            # output_function=T.nnet.sigmoid
            # #training sigmoid autoencoders greedily
            # for val in hidden:
            #     self.encoders.append(Autoencoder.AutoEncoder(input_data, val, activation_function, output_function))
            #     self.encoders[i].train(n_epochs=max_iterations, mini_batch_size=1, learning_rate=0.05)
            #     input_data = self.encoders[i].get_hidden(input_data)
            #     i += 1
            # #training final softmax autoencoder layer
            # activation_function = T.nnet.softmax
            # self.encoders.append(Autoencoder.AutoEncoder(input_data, num_topics, activation_function, output_function))
            # self.encoders[i].train(n_epochs=max_iterations, mini_batch_size=1, learning_rate=0.05)
        elif self.architecture == 'dbn-auto':
            logger.debug('dbn-auto')
            if not input_data:
                input_data = self.features.toarray()
            frequency_weights = float(input_data.shape[0])/input_data.sum(axis=0) * 1000
            numpy.random.shuffle(input_data)
            # input_data = input_data[1:5,:]
            self.dbn = train_DBN(input_data, finetune_lr=0.1, pretraining_epochs=max_iterations,
                                     pretrain_lr=0.1, k=1, training_epochs=max_iterations, batch_size=100,
                                     hidden_layers_sizes=hidden, softmax_size=num_topics, sparsity=sparsity,
                                     selectivity=selectivity, frequency_weights=frequency_weights)
        else:
            logger.error("NO SUCH ARCHITECTURE!!!!")

    def print_topics(self, index, num_words, word_to_idx=None, filepath=None):

        """ Returns a number of words (as set by num_words) that correspond to a topic (defined by index) """

        # activate one of the softmax units
        t = numpy.zeros((1, self.num_topics))
        t[0][index] = 1000
        activations = self.reconstruct_given_hidden(t)[0]
        topics = {}
        i = 0
        # map vector to the correct words
        if not word_to_idx:
            word_to_idx = self.vectorizer.get_feature_names()
        for a in activations:
            topics[word_to_idx[i]] = a
            i += 1
        sorted_topics = sorted(topics.items(), key=operator.itemgetter(1), reverse=True)
        logger.debug("Topic %s", index)
        topic = []
        for j in range(num_words):
            logger.debug("%s:%s", sorted_topics[j][0], sorted_topics[j][1])
            topic.append(sorted_topics[j][0].__str__())
        if filepath is not None:
            with open(filepath, 'a') as f:
                s = "Topic " + index.__str__()
                print>> f, s
                j = 0
                for j in range(num_words):
                    s = sorted_topics[j][0].__str__() + ":" + sorted_topics[j][1].__str__()
                    print>> f, s
        return topic

    def get_topic(self, index, num_words):

        """ Returns a number of words (as set by num_words) that correspond to a topic (defined by index) """

        # activate one of the softmax units
        t = numpy.zeros((1, self.num_topics))
        t[0][index] = 1
        activations = self.reconstruct_given_hidden(t)[0]
        topics = {}
        i = 0
        # map vector to the correct words
        words = self.vectorizer.get_feature_names()
        for a in activations:
            topics[words[i]] = a
            i += 1
        sorted_topics = sorted(topics.items(), key=operator.itemgetter(1), reverse=True)
        topic = []
        for j in range(num_words):
            topic.append(sorted_topics[j][0].__str__())
        return topic

    def reconstruct_given_hidden(self, hidden):

        """ Computes output of network (i.e. reconstructed input) given the hidden activations of the network"""

        if self.architecture == 'stacked-auto':
            # recursively reconstruct the probable input given the activations of the hidden softmax layer
            for encoder in reversed(self.encoders):
                hidden = encoder.outputgivenhidden(hidden)
        elif self.architecture == 'dbn-auto':
            hidden = self.dbn.decode(hidden)
        return hidden

    def feed_forward(self, input_data):

        """ Computes the hidden layer activations given the input to the network"""

        if self.architecture == 'stacked-auto':
            # recursively compute the output of the stacked autoencoder
            for encoder in self.encoders:
                input_data = encoder.get_hidden(input_data)
        elif self.architecture == 'dbn-auto':
            input_data = self.dbn.encode(input_data)
        return input_data

    def get_topic_distribution(self, sentence):

        """ Returns the topic distribution of a single sentence """

        sent_tfidf = self.vectorizer.transform([" ".join(pr.stem_doc(sentence.split(' ')))])
        dist = self.feed_forward(sent_tfidf.toarray())[0]
        return dist

    def get_topic_distribution_batch(self, sentences):

        """ Returns the topic distribution of a batch of sentences """
        stemmed_sentences = [" ".join(pr.stem_doc(sentence.split(' '))) for sentence in sentences]
        sent_tfidf = self.vectorizer.transform(stemmed_sentences)
        dist = self.feed_forward(sent_tfidf.toarray())
        return dist

    def save(self, filepath):

        """Saves the model to the designated file path"""

        with open(filepath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def getIntrinisicCoherence(self, topics=None):

        """Returns the average coherence of all the topics generated by the model"""

        if not topics:
            topics = []
            for x in range(0, self.num_topics):
                topics.append(self.get_topic(x, 20))
                # topics.append(self.print_topics(x,20))
        coherence = 0
        for topic in topics:
            for pair in itertools.combinations(topic, 2):
                coherence += self.logProbability(pair)
        return float(coherence) / len(topics)

    def logProbability(self, pair):
        # num_pair, num_uni_one, num_uni_two = self.numUniBigrams(pair)
        # num_pair = float(num_pair)
        num_pair = float(self.numBigrams(pair) + 1)
        num_uni_one = float(self.numUnigrams(pair[0]))
        num_uni_two = float(self.numUnigrams(pair[1]))
        return (math.log10(num_pair / num_uni_one) + math.log10(num_pair / num_uni_two)) / 2

    def numUniBigrams(self, pair):
        countunione = 0
        countunitwo = 0
        countbi = 0
        for d in self.documents:
            splitdoc = d.split()
            if pair[0] in splitdoc and pair[1] in splitdoc:
                countbi += 1
            if pair[0] in d.split():
                countunione += 1
            if pair[1] in d.split():
                countunitwo += 1
        return [countunione, countunitwo, countbi + 1]

    def numBigrams(self, pair):

        """ Counts the number of times a pair of words appear in an entire corpus """

        count = 0
        for d in self.documents:
            splitdoc = d.split()
            if pair[0] in splitdoc and pair[1] in splitdoc:
                count += 1
        return count

    def numUnigrams(self, word):

        """ Counts the number of times a word appears in the entire corpus """

        count = 0
        for d in self.documents:
            if word in d.split():
                count += 1
        return count

    def getName(self):

        """ Returns paramters of the model concatenated as a string"""

        return str(self.hidden) + ',' + str(self.df) + ',' + str(self.max_iterations) + ',' + str(
            self.sparsity) + ',' + str(self.num_topics) + ',' + self.architecture

    @staticmethod
    def load(filepath):
        dl = None
        with open(filepath, 'rb') as input:
            dl = pickle.load(input, encoding='latin1') #hacky 2to3
        return dl

    def sigmoid(self, z):
        s = 1.0 / (1.0 + numpy.exp(-1.0 * z))
        return s
