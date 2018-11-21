"""
Modified from http://deeplearning.net/tutorial/code/DBN.py
Implementation of a DBN composed sparse, selective RBM units
"""
import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

from .mlp import HiddenLayer
from .rbm import RBM
from .softmax_rbm import Softmax_RBM


class DBN(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=None, softmax_size=25, sparsity=0.03,
                 selectivity=0.03, frequency_weights=None):
        """
        A DBN composed of sparse, selective RBM units
        
        Parameters:
        
        numpy_rng: numpy random number generator
        theano_rng: theano random number generator
        n_ins: number of input neurons
        hidden_layer_sizes: list of number of neurons in each hidden layer
        softmax_size: number of neurons in the bottleneck softmax layer
        sparsity: sparsity constant for RBMs
        """
        if not hidden_layers_sizes:
            hidden_layers_sizes = [500, 500]
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.frequency_weights = frequency_weights

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')

        # allocate symbolic variable for freq weights
        self.freq_weight_sym = T.vector('fr_weight')

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
                input_layer = True
            else:
                layer_input = self.sigmoid_layers[-1].output
                input_layer = False

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid,
                                        out_activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer. NOTE: BIASES ARE SHARED
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b1,
                            vbias=sigmoid_layer.b2,
                            sparsity_target=sparsity,
                            selectivity_target=selectivity)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a softmax layer on top of the autoencoder
        self.softmaxLayer = HiddenLayer(rng=numpy_rng,
                                        input=self.sigmoid_layers[-1].output,
                                        n_in=hidden_layers_sizes[-1],
                                        n_out=softmax_size,
                                        activation=T.nnet.softmax,
                                        out_activation=self.sigmoid_layers[-1].activation)

        self.params.extend(self.softmaxLayer.params)
        softmax_rbm_layer = Softmax_RBM(numpy_rng=numpy_rng,
                                        theano_rng=theano_rng,
                                        input=self.sigmoid_layers[-1].output,
                                        n_visible=hidden_layers_sizes[-1],
                                        n_hidden=softmax_size,
                                        W=self.softmaxLayer.W,
                                        hbias=self.softmaxLayer.b1,
                                        vbias=self.softmaxLayer.b2)
        self.rbm_layers.append(softmax_rbm_layer)

        # compute the cost for second phase of training, defined as 
        # cross-entropy
        self.finetune_cost = self.cost(frequency_weights=frequency_weights)

    def reconstruct_input(self, input_data):

        """Reconstructs the input_data """

        hidden = self.encode(input_data)
        return self.decode(hidden)

    def encode(self, input_data):

        """Returns the output of the bottle-neck softlayer given the input to the network i.e. encodes the input"""

        x = T.dmatrix('x')
        hidden = 0
        for i in range(0, self.n_layers):
            l = self.sigmoid_layers[i]
            if i == 0:
                hidden = l.activation(T.dot(x, l.W) + l.b1)
            else:
                hidden = l.activation(T.dot(hidden, l.W) + l.b1)
        hidden = T.nnet.softmax(T.dot(hidden, self.softmaxLayer.W) + self.softmaxLayer.b1)
        transformed_data = theano.function(inputs=[x], outputs=[hidden])
        return transformed_data(input_data)[0]

    def decode(self, hidden):

        """Returns the output of the network given the output of the bottle-neck softmax layer i.e. decodes the hidden activations"""

        h = T.dmatrix('h')
        l = self.softmaxLayer
        input = l.out_activation(T.dot(h, l.W.T) + l.b2)
        for i in range(1, self.n_layers + 1):
            l = self.sigmoid_layers[-i]
            input = l.activation(T.dot(input, l.W.T) + l.b2)
        transformed_data = theano.function(inputs=[h], outputs=[input])
        return transformed_data(hidden)[0]

    def cost(self, frequency_weights=None):

        """Computes cross-entropy cost to finetune the RBM-Autoencoder"""

        # Computing output of network
        h = self.softmaxLayer.output
        l = self.softmaxLayer
        reconstructed_input = l.out_activation(T.dot(h, l.W.T) + l.b2)
        for i in range(1, self.n_layers + 1):
            l = self.sigmoid_layers[-i]
            reconstructed_input = l.out_activation(T.dot(reconstructed_input, l.W.T) + l.b2)
        # L = T.sum(((self.x - reconstructed_input) ** 2), axis=1)
        # Return mean cross-entropy
        L = self.x * T.log(reconstructed_input) + (1 - self.x) * T.log(1 - reconstructed_input)
        L = -T.sum(L, axis=0)
        if frequency_weights is not None:
            L = self.freq_weight_sym * L
        cost = L.mean()
        return cost

    def pretraining_functions(self, data, batch_size, k):
        """Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.
        """

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = data.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: data[batch_begin:batch_end, :]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, data, batch_size, learning_rate):
        """Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set """

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: data[index * batch_size: (index + 1) * batch_size, :],
                self.freq_weight_sym: self.frequency_weights
            }
        )

        return train_fn


def train_DBN(data, finetune_lr=0.01, pretraining_epochs=100, pretrain_lr=0.1, k=1, training_epochs=10, batch_size=10,
              hidden_layers_sizes=None, softmax_size=2, sparsity=0.03, selectivity=0.03, frequency_weights=None):
    if not hidden_layers_sizes:
        hidden_layers_sizes = [6]
    print("train: " + str(selectivity))
    print("train: " + str(sparsity))
    data = theano.shared(name='data', value=np.asarray(data,
                                                       dtype=theano.config.floatX), borrow=True)
    # compute number of minibatches for training, validation and testing
    n_train_batches = data.get_value(borrow=True).shape[0] / batch_size
    n_ins = data.get_value(borrow=True).shape[1]
    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes,
              softmax_size=softmax_size, sparsity=sparsity,
              selectivity=selectivity, frequency_weights=frequency_weights)
    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    # print('... getting the pretraining functions')
    # pretraining_fns = dbn.pretraining_functions(data=data,
    #                                             batch_size=batch_size,
    #                                             k=k)

    # print('... pre-training the model')
    # start_time = time.clock()
    # # Pre-train layer-wise
    # for i in xrange(len(dbn.rbm_layers)):
    #     # go through pretraining epochs
    #     for epoch in xrange(pretraining_epochs):
    #         # go through the training set
    #         c = []
    #         for batch_index in xrange(n_train_batches):
    #             c.append(pretraining_fns[i](index=batch_index,
    #                                         lr=pretrain_lr))
    #         print('Pre-training layer %i, epoch %d, cost ' % (i, epoch),)
    #         print(numpy.mean(c))

    # end_time = time.clock()
    # print ('Pre-Training took %f sec' % (end_time - start_time))
    # print ('Pre-Training took %f sec per epoch' % ((end_time - start_time) / (epoch + 1)))
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn = dbn.build_finetune_functions(
        data=data,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    print(softmax_size)
    print('... finetuning the model')
    start_time = time.clock()
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_fn(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c))

    end_time = time.clock()
    print ('Fine-tuning took %f sec' % (end_time - start_time))
    print ('Fine-tuning took %f sec per epoch' % ((end_time - start_time) / (epoch + 1)))

    return dbn

# s = np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1], [1,1,1,1,1,1],[1,1,1,1,1,1]])
# s= np.random.rand(1000,6)
# print s
# dbn = train_DBN(data=s)
# from revup.core.TopicModelling.dlmodel import SAmodel
# x = SAmodel(hidden=[50], max_iterations=20, num_topics=20, file_path='/mnt/c/Users/giris/Downloads/spainv1.txt')