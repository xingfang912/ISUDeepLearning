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
import matplotlib.pyplot as plt
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.nnet import conv
import theano.printing
from theano.tensor.signal.pool import pool_2d

from ISUDeepLearning.DataMungingUtil import load_dataset

from ISUDeepLearning.DNN_functions import ReLU, Sigmoid, Tanh, HiddenLayer, dropout_from_layer, DropoutHiddenLayer


def load_data(path, random_seed=None):
    num_classes, (train_set, valid_set, test_set) = load_dataset(path, random_seed=random_seed)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return num_classes, rval


class CPLayer(object):
    """CP Layer of the tiny VGG 
       C is convolution
       P is pooling, which can be turned off resulting a pure convolutional layer, C     
    """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), pooling=False, use_bias=False):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True,
            name = 'W'
        )
        
        if use_bias:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name='b')

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        
        # Applying activations
        act_out = T.maximum(conv_out,0)


        if pooling:
            # downsample each feature map individually, using maxpooling
            pooled_out = pool_2d(
                input=act_out,
                ds=poolsize,
                ignore_border=True
            )

            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
            if use_bias:
                self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
            else:
                self.output = pooled_out
        else:
            if use_bias:
                self.output = act_out + self.b.dimshuffle('x', 0, 'x', 'x')
            else:
                self.output = act_out

        # store parameters of this layer
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

        # keep track of model input
        self.input = input


# Stuff to add: 
# N best list with confidence scores. (So we can get Rank-1 Rank-2, etc.)
# Log the training error, validation error, and store in external file.
# Log weight change map over time.
# List of most often misclassified samples. For each sample, how hard was it to learn?
# Feature that detects overfitting and automatically terminates training,
#   or in the future adjusts the learning rate.
def test_net(
        classifier,
        num_classes,
        learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        x,
        mom_params,
        dropout,
        results_file_name,
        dataset,
        use_bias,
        random_seed,
        decay=True,
        momentum=True,
        L2=True,
        plot = False):

    [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)] = dataset
    
    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    rng = np.random.RandomState(random_seed)

    # Build the expresson for the cost function.
    if L2:
        lamb = 0.00000001
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)
        if use_bias:
            cost += lamb * sum([(classifier.params[i]**2).sum() for i in range(0,len(classifier.params),2)])/2*batch_size
            dropout_cost += lamb * sum([(classifier.params[i]**2).sum() for i in range(0,len(classifier.params),2)])/2*batch_size
        else:
            cost += lamb *sum([(param**2).sum() for param in classifier.params])/2*batch_size
            dropout_cost += lamb *sum([(param**2).sum() for param in classifier.params])/2*batch_size
    else:
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]},
            on_unused_input='ignore')
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]},
            on_unused_input='ignore')
    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    if momentum:
        print >> sys.stderr, ("Using momentum")
        # ... and allocate mmeory for momentum'd versions of the gradient
        gparams_mom = []
        for param in classifier.params:
            gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                dtype=theano.config.floatX))
            gparams_mom.append(gparam_mom)
    
        # Compute momentum for the current epoch
        mom = ifelse(epoch < mom_epoch_interval,
                mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
                mom_end)
    
        
        # Update the step direction using momentum
        updates = OrderedDict()
        for gparam_mom, gparam in zip(gparams_mom, gparams):
            # Misha Denil's original version
            #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
          
            # change the update rule to match Hinton's dropout paper
            updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam
    
        # ... and take a step along that direction
        for param, gparam_mom in zip(classifier.params, gparams_mom):
            # Misha Denil's original version
            #stepped_param = param - learning_rate * updates[gparam_mom]
            
            # since we have included learning_rate in gparam_mom, we don't need it
            # here
            stepped_param = param + updates[gparam_mom]
    
            # This is a silly hack to constrain the norms of the rows of the weight
            # matrices.  This just checks if there are two dimensions to the
            # parameter and constrains it if so... maybe this is a bit silly but it
            # should work for now.
            if param.get_value(borrow=True).ndim == 2:
                #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
                #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
                #updates[param] = stepped_param * scale
                
                # constrain the norms of the COLUMNs of the weight, according to
                # https://github.com/BVLC/caffe/issues/109
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param

    else:
        
        if L2:
            print >> sys.stderr, ("Using gradient decent with L2 regularization")
            updates = [
            (param_i, param_i - learning_rate * (grad_i + lamb*param_i/batch_size))
            for param_i, grad_i in zip(classifier.params, gparams)
            ]
        else:
            print >> sys.stderr, ("Using gradient decent")
            updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(classifier.params, gparams)
            ]
    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch,index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]},
            on_unused_input='ignore')
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

#    best_params = None
    best_validation_errors = np.inf
    best_test_score = np.inf
    best_iter_valid = 0
    best_iter_test = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

#    results_file = open(results_file_name, 'wb')

    plot_training = []
    plot_valid = []
    plot_test = []

    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        minibatch_avg_cost = 0
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost += train_model(epoch_counter, minibatch_index)
        
        plot_training.append(minibatch_avg_cost/n_train_batches)

        # Compute loss on validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_errors = np.mean(validation_losses)
        
        plot_valid.append(this_validation_errors)

        # Report and save progress.
        print "epoch {}, validation error {}%, learning_rate={}{}".format(
                epoch_counter, this_validation_errors*100,
                learning_rate.get_value(borrow=True),
                " **" if this_validation_errors < best_validation_errors else "")
        if this_validation_errors < best_validation_errors:
            best_iter_valid = epoch_counter

        best_validation_errors = min(best_validation_errors, this_validation_errors)
#        results_file.write("{0}\n".format(this_validation_errors))
#        results_file.flush()
                
        # test it on the test set
        test_losses = [
            test_model(i)
            for i in xrange(n_test_batches)
        ]
        test_score = np.mean(test_losses)
        
        plot_test.append(test_score)
        
        print(('     epoch %i, test error of '
               'best model %f %%') %
              (epoch_counter, test_score * 100.))
        if test_score < best_test_score:
            best_test_score = test_score
            best_iter_test = epoch_counter
            print >> sys.stderr,('     Current best test score: '+str(best_test_score*100)+'%')
        
        if decay:
            new_learning_rate = decay_learning_rate()

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at epoch %i') %
          (best_validation_errors * 100., best_iter_valid))
    print >> sys.stderr, (('Best test score of %f %% '
           'obtained at epoch %i') %
          (best_test_score * 100., best_iter_test))      
      
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    if plot:
        Epoch = np.arange(1,n_epochs+1)
        plt.subplot(3, 1, 1)
        plt.plot(Epoch, plot_training)
        plt.grid(axis="y")
        plt.ylabel('Training Error',fontsize=14)
        
        plt.subplot(3, 1, 2)
        plt.plot(Epoch, plot_valid,color="g")
        plt.grid(axis="y")
        plt.xlabel('Iteration',fontsize=18)
        plt.ylabel('Validation Error',fontsize=14)
        
        plt.subplot(3, 1, 3)
        plt.plot(Epoch, plot_test,color="r")
        plt.grid(axis="y")
        plt.xlabel('Iteration',fontsize=18)
        plt.ylabel('Testing Error',fontsize=14)

    return (plot_training, plot_valid, plot_test)
