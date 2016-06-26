"""
This tutorial introduces the tiny VGG architecture (C-CP-C-CP-C-CP-FC)
    Xing Fang (School of Information Technology, Illinois State University)
        and 
    Ryan Dellana (Department of Computer Science, North Carolina A&T State University)

Part of this program uses the code found at: https://github.com/mdenil/dropout

Date: May 27, 2016
"""
import numpy as np
import os, sys
import time

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.nnet import conv
import theano.printing
from theano.tensor.signal.pool import pool_2d

from logistic_sgd import LogisticRegression

from DBN_functions import *

from CNNLib import *


class TVGG(object):
    """A tiny VGG net.
    """
    def __init__(self,
            rng,
            input,
            nkerns,
            dropout_rates,
            mlp_layer_sizes,
            activations,
            batch_size,
            use_bias=True):


        layer0_input = input
        # Construct the first convolutional relu layer:
        # filtering reduces the image size to (224-3+1 , 224-3+1) = (222, 222)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 222, 222)
        layer0 = CPLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 224, 224),
            filter_shape=(nkerns[0], 3, 3, 3),
        )
        # Construct the second convolutional relu layer
        # filtering reduces the image size to (222-3+1, 222-3+1) = (220, 220)
        # maxpooling reduces this further to (220/2, 220/2) = (110, 110)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 110, 110)
        layer1 = CPLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 222, 222),
            filter_shape=(nkerns[1], nkerns[0], 3, 3),
            pooling=True
        )
        
        # Construct the third convolutional relu layer:
        # filtering reduces the image size to (110-3+1 , 110-3+1) = (108, 108)
        # 4D output tensor is thus of shape (batch_size, nkerns[2], 108, 108)
        layer2 = CPLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, nkerns[1], 110, 110),
            filter_shape=(nkerns[2], nkerns[1], 3, 3),
        )

        # Construct the fourth convolutional relu layer
        # filtering reduces the image size to (108-3+1, 108-3+1) = (106, 106)
        # maxpooling reduces this further to (106/2, 106/2) = (53, 53)
        # 4D output tensor is thus of shape (batch_size, nkerns[3], 53, 53)
        layer3 = CPLayer(
            rng,
            input=layer2.output,
            image_shape=(batch_size, nkerns[2], 108, 108),
            filter_shape=(nkerns[3], nkerns[2], 3, 3),
            pooling=True
        )

        # Construct the fifth convolutional relu layer
        # filtering reduces the image size to (53-4+1, 53-4+1) = (50, 50)
        # 4D output tensor is thus of shape (batch_size, nkerns[4], 50, 50)
        layer4 = CPLayer(
            rng,
            input=layer3.output,
            image_shape=(batch_size, nkerns[3], 53, 53),
            filter_shape=(nkerns[4], nkerns[3], 4, 4),
        )

        # Construct the sixth convolutional relu layer
        # filtering reduces the image size to (50-3+1, 50-3+1) = (48, 48)
        # maxpooling reduces this further to (48/2, 48/2) = (24, 24)
        # 4D output tensor is thus of shape (batch_size, nkerns[5], 24, 24)
        layer5 = CPLayer(
            rng,
            input=layer4.output,
            image_shape=(batch_size, nkerns[4], 50, 50),
            filter_shape=(nkerns[5], nkerns[4], 3, 3),
            pooling=True
        )

        # Set up all the hidden layers
        weight_matrix_sizes = zip(mlp_layer_sizes, mlp_layer_sizes[1:])
        self.mlp_layers = []
        self.mlp_dropout_layers = []
        next_layer_input = layer5.output.flatten(2)

        # dropout the input
        next_dropout_layer_input = dropout_from_layer(rng, layer5.output.flatten(2), p=dropout_rates[0])
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter + 1])
            self.mlp_dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.mlp_layers.append(next_layer)
            next_layer_input = next_layer.output
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(rng, input=next_dropout_layer_input, n_in=n_in, n_out=n_out, use_bias=use_bias)
        self.mlp_dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(rng, input=next_layer_input,
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out, use_bias=use_bias)
        self.mlp_layers.append(output_layer)


        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.mlp_dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.mlp_dropout_layers[-1].errors

        self.negative_log_likelihood = self.mlp_layers[-1].negative_log_likelihood
        self.errors = self.mlp_layers[-1].errors


        # Grab all the parameters together.
        self.params = [ param for layer in self.mlp_dropout_layers for param in layer.params ] + \
                      layer5.params + layer4.params + layer3.params + layer2.params + \
                      layer1.params + layer0.params


if __name__ == '__main__':
    
    # set the random seed to enable reproduciable results
    # It is used for initializing the weight matrices
    # and generating the dropout masks for each mini-batch
    random_seed = 23455
    initial_learning_rate = 0.01
    learning_rate_decay = 0.998
    squared_filter_length_limit = 15.0
    n_epochs = 70
    batch_size = None
    nkerns = [64, 64, 128, 128, 256, 256]
    #24 is the last feature map size
    #feature map sizes = [224,222,110,108,53,50,24]
    mlp_layer_sizes = [ 256*24*24, 500, -1 ] # -1 = num classes. not known initially.
    # activation functions for each layer
    # For this demo, we don't need to set the activation functions for the 
    # on top layer, since it is always 10-way Softmax
    activations = [ Tanh ]
    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.99
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 100
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}    
    dataset = '/home/ryan/datasets/Cropped_Yale_Subset_Periocular_resized_pad_224'

#    if len(sys.argv) < 2:
#        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
#        exit(1)
#
#    elif sys.argv[1] == "dropout":
#        dropout = True
#        results_file_name = "results_dropout.txt"
#
#    elif sys.argv[1] == "backprop":
#        dropout = False
#        results_file_name = "results_backprop.txt"
#
#    else:
#        print "I don't know how to '{0}'".format(sys.argv[1])
#        exit(1)
    dropout = False
    if dropout:
        # dropout rate for each layer
        dropout_rates = [ 0.2, 0.5] 
    else:
        dropout_rates = [ 0, 0]    
    
    use_bias = False
    results_file_name = None

    num_classes, dataset_ = load_data(dataset, random_seed=random_seed)
    # [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

    assert len(mlp_layer_sizes) - 1 == len(dropout_rates)
    mlp_layer_sizes[2] = num_classes # set num_classes based on dataset.

    if batch_size is None:
        batch_size = num_classes
    
    x = T.tensor4('x')  # the data is presented as rasterized images

    learning_rate = theano.shared(np.asarray(initial_learning_rate, dtype=theano.config.floatX))

    classifier = TVGG(rng = np.random.RandomState(seed=random_seed), 
                      input = x.reshape((batch_size, 3, 224, 224)),
                      nkerns = nkerns,
                      dropout_rates = dropout_rates,
                      mlp_layer_sizes = mlp_layer_sizes,
                      activations = activations,
                      batch_size = batch_size,
                      use_bias = use_bias)

    test_net(classifier = classifier,
             num_classes = num_classes,
             learning_rate = learning_rate,
             learning_rate_decay = learning_rate_decay,
             squared_filter_length_limit = squared_filter_length_limit,
             n_epochs = n_epochs,
             batch_size = batch_size,
             x = x,
             mom_params = mom_params,
             dropout = dropout,
             dataset = dataset_,
             use_bias = use_bias,
             results_file_name = results_file_name,
             random_seed = random_seed,
             decay = False,
             momentum = False,
             L2 = False,
             plot = False)
