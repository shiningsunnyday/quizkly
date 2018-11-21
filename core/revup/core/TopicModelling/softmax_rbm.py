# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 17:31:17 2015

@author: giris_000
"""

import theano
import theano.tensor as T

from .rbm import RBM


class Softmax_RBM(RBM):
    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, numpy_rng=None,
                 theano_rng=None, selectivity_target=0.2, selectivity_param=0, wl1_param=1, wl2_param=1):

        RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden, W=W, hbias=hbias,
                     vbias=vbias, numpy_rng=numpy_rng, theano_rng=theano_rng, selectivity_target=selectivity_target,
                     sparsity_target=selectivity_target)
        self.selectivity_target = selectivity_target
        self.selectivity_param = selectivity_param
        self.wl1_param = wl1_param
        self.wl2_param = wl2_param

    def propup(self, vis):
        """This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-softmax activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        """
        pre_softmax_activation = T.dot(vis, self.W) + self.hbias
        return [pre_softmax_activation, T.nnet.softmax(pre_softmax_activation)]

    def free_energy(self, v_sample):
        """ Function to compute the free energy """
        vbias_term = T.dot(v_sample, self.vbias)

        wx_b = T.dot(v_sample, self.W) + self.hbias
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_reconstruction_cost(self, updates, pre_softmax_nv):
        """
        Approximation to the reconstruction error
        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.softmax(pre_softmax_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.softmax(pre_softmax_nv)),
                axis=1
            )
        )

        return cross_entropy

    def sample_h_given_v(self, v0_sample):
        """ This function infers state of hidden units given visible units """
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_mean]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_softmax_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_softmax_nvs,
                nv_means,
                nv_samples,
                pre_softmax_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        # selectivity and sparsity cost
        sp_cost = (self.selectivity_param * self.selectivity_term()) + (self.selectivity_param * self.sparsity_term())
        reg_sp_cost = sp_cost + self.get_hbias_reg()
        sp_grad = T.grad(reg_sp_cost, self.hbias)
        # regularise vbias
        decay_vgrad = T.grad(self.get_vbias_reg(), self.vbias)
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            if param == self.hbias:
                updates[param] = param - gparam * T.cast(
                    lr,
                    dtype=theano.config.floatX
                ) - sp_grad * T.cast(
                    lr,
                    dtype=theano.config.floatX
                )
            elif param == self.vbias:
                updates[param] = param - gparam * T.cast(
                    lr,
                    dtype=theano.config.floatX
                ) - decay_vgrad * T.cast(
                    lr,
                    dtype=theano.config.floatX
                )
            else:
                updates[param] = param - gparam * T.cast(
                    lr,
                    dtype=theano.config.floatX
                )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_softmax_nvs[-1])
        return monitoring_cost, updates

    def sample_v_given_h(self, h0_sample):
        """ This function infers state of visible units given hidden units """
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """ This function implements one step of Gibbs sampling,
            starting from the hidden state"""
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """ This function implements one step of Gibbs sampling,
            starting from the visible state"""
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def funny_cost(self):
        h = self.propup(self.input)[1]
        return T.median(T.mean(h, axis=0))
