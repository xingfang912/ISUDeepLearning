"""
This tutorial introduces a convolutional neural network (C-CP-FC)
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

from ISUDeepLearning.logistic_sgd import LogisticRegression
from ISUDeepLearning.DNN_functions import ReLU, Sigmoid, Tanh, HiddenLayer, dropout_from_layer, DropoutHiddenLayer
from ISUDeepLearning.CNNLib import CPLayer, load_data, test_net

class CNN(object):

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

        # Set up all the hidden layers
        weight_matrix_sizes = zip(mlp_layer_sizes, mlp_layer_sizes[1:])
        self.mlp_layers = []
        self.mlp_dropout_layers = []
        next_layer_input = layer1.output.flatten(2)

        # dropout the input
        next_dropout_layer_input = dropout_from_layer(rng, layer1.output.flatten(2), p=dropout_rates[0])
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
        self.dropout_output_layer = dropout_output_layer

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(rng, input=next_layer_input,
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out, use_bias=use_bias)
        self.mlp_layers.append(output_layer)
        self.output_layer = output_layer

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_output_layer.negative_log_likelihood
        self.dropout_errors = self.dropout_output_layer.errors

        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors

        # Grab all the parameters together.
        self.params = [ param for layer in self.mlp_dropout_layers for param in layer.params ] + \
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
    nkerns = [64, 64]

    #feature map sizes = [224,222,110]
    mlp_layer_sizes = [ nkerns[-1]*110*110, 500, -1 ] # -1 = num classes. not known initially.

    # activation functions for the fully connected (FC) layer
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

    #dataset = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_40s_51c_32t_4v_4tst_resized_pad_224'
    #dataset = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_50s_27c_40t_5v_5tst_resized_pad_224'
    dataset = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_10s_98c_8t_1v_1tst_resized_pad_224'

    dropout = False

    if dropout:
        # dropout rate for the two layers: 
	# first layer is the input layer of the MLP
	# sencond layer is the hidden layer of the MLP
        dropout_rates = [ 0.2, 0.5 ] 
    else:
        dropout_rates = [ 0, 0 ]    
    
    use_bias = False
    results_file_name = None

    num_classes, dataset_ = load_data(dataset, random_seed=random_seed)

    assert len(mlp_layer_sizes) - 1 == len(dropout_rates)
    mlp_layer_sizes[2] = num_classes # set num_classes based on dataset.

    if batch_size is None:
        batch_size = num_classes
    
    x = T.tensor4('x')  # the data is presented as rasterized images

    learning_rate = theano.shared(np.asarray(initial_learning_rate, dtype=theano.config.floatX))

    classifier = CNN(rng = np.random.RandomState(seed=random_seed), 
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
             L2 = True,
             plot = False)
