"""
This tutorial introduces the tiny VGG architecture (C-CP-C-CP-C-CP-FC)
    Xing Fang (School of Information Technology, Illinois State University)
        and 
    Ryan Dellana (Department of Computer Science, North Carolina A&T State University)

Part of this program uses the code found at: https://github.com/mdenil/dropout

Date: May 27, 2016
"""

import theano
import theano.tensor as T
import numpy as np


##################################
## Various activation functions ##
##################################
#### rectified linear unit
def ReLU(x):
    y = T.nnet.relu(x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None, Type = 'Xavier',
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            if Type == 'Xavier':
                W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX)
                W = theano.shared(value=W_values, name='W')
            else:
                W_values = np.asarray(0.01 * rng.standard_normal(
                    size=(n_in, n_out)), dtype=theano.config.floatX)
                W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = T.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = dropout_from_layer(rng, self.output, p=dropout_rate)

