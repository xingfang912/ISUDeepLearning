"""This tutorial introduces the tiny VGG architecture (CR-CR-P-CR-CR-P-CR-CR-P-FC)
    Xing Fang and Ryan Dellana 
    Department of Computer Science 
    North Carolina A&T State University
"""
import os
import sys
import timeit
import pylab
import numpy as np

import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv
import DataMungingUtil
path = '/home/ryan/Documents/CNN_dellana' # TODO Add this to python path in .bashrc
sys.path.append(path)
from logistic_sgd import LogisticRegression

def load_data(path):
    num_classes, (train_set, valid_set, test_set) = DataMungingUtil.load_dataset(path)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
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

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class CRPLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), pooling=False):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
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
            borrow=True
        )
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        #relu_out = T.maximum(conv_out,0)  # RELU input feature maps
        relu_out = T.nnet.relu(conv_out)
        if pooling:
            # downsample each feature map individually, using maxpooling
            pooled_out = pool_2d(
                input=relu_out,
                ds=poolsize,
                ignore_border=True
            )
            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
            # self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = pooled_out+self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            self.output = relu_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]  # store parameters of this layer
        self.input = input              # keep track of model input


def evaluate_vgg(learning_rate=0.05, n_epochs=70,
                    dataset_path='/home/ryan/datasets/Cropped_Yale_Subset_Periocular_resized_pad_224',
                    nkerns=[64, 64, 128, 128, 256, 256], batch_size=None):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    rng = np.random.RandomState(23455)
    num_classes, datasets = load_data(dataset_path)
    train_set_x, train_set_y = datasets[0] # x = sample, y = label
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    if batch_size is None:
        batch_size = num_classes
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size  #320/20 = 16
    n_valid_batches /= batch_size  #40/20 = 2
    n_test_batches /= batch_size   #40/20 = 2
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.tensor4('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    print '... building the model'  # vvv BUILD ACTUAL MODEL
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 224, 224))
    # Construct the first convolutional relu layer:
    # filtering reduces the image size to (224-3+1 , 224-3+1) = (222, 222)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 222, 222)
    layer0 = CRPLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 224, 224),
        filter_shape=(nkerns[0], 3, 3, 3),
    )
    # Construct the second convolutional relu layer
    # filtering reduces the image size to (222-3+1, 222-3+1) = (220, 220)
    # maxpooling reduces this further to (220/2, 220/2) = (110, 110)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 110, 110)
    layer1 = CRPLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 222, 222),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        pooling=True
    )
    # Construct the third convolutional relu layer:
    # filtering reduces the image size to (110-3+1 , 110-3+1) = (108, 108)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 108, 108)
    layer2 = CRPLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 110, 110),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
    )
    # Construct the fourth convolutional relu layer
    # filtering reduces the image size to (108-3+1, 108-3+1) = (106, 106)
    # maxpooling reduces this further to (106/2, 106/2) = (53, 53)
    # 4D output tensor is thus of shape (batch_size, nkerns[3], 53, 53)
    layer3 = CRPLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 108, 108),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        pooling=True
    )
    # Construct the fifth convolutional relu layer
    # filtering reduces the image size to (53-4+1, 53-4+1) = (50, 50)
    # 4D output tensor is thus of shape (batch_size, nkerns[4], 50, 50)
    layer4 = CRPLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], 53, 53),
        filter_shape=(nkerns[4], nkerns[3], 4, 4),
    )
    # Construct the sixth convolutional relu layer
    # filtering reduces the image size to (50-3+1, 50-3+1) = (48, 48)
    # maxpooling reduces this further to (48/2, 48/2) = (24, 24)
    # 4D output tensor is thus of shape (batch_size, nkerns[5], 24, 24)
    layer5 = CRPLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, nkerns[4], 50, 50),
        filter_shape=(nkerns[5], nkerns[4], 3, 3),
        pooling=True
    )
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 10 * 10),
    # or (500, 50 * 10 * 10) = (500, 5000) with the default values.
    layer6_input = layer5.output.flatten(2)
    # construct a fully-connected sigmoidal layer
    layer6 = HiddenLayer(
        rng,
        input=layer6_input,
        n_in=nkerns[5] * 24 * 24,
        n_out=500,
        activation=T.tanh
    )
    # classify the values of the fully-connected sigmoidal layer
    layer7 = LogisticRegression(rng=np.random.RandomState(42), input=layer6.output, n_in=500, n_out=num_classes)
    # the cost we minimize during training is the NLL of the model
    cost = layer7.negative_log_likelihood(y)
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a list of all model parameters to be fit by gradient descent
    params = layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1
    print '... training' # vvv TRAIN MODEL
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False
    validationError = []
    while (epoch < n_epochs) and (not done_looping):  
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
#            if iter % 10 == 0:
            print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            print cost_ij
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                #For validation error plot
                validationError.append(this_validation_loss * 100)       
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches) ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))   
    x_axis = np.arange(0,n_epochs,1)
    pylab.plot(x_axis,validationError)
    pylab.show()

if __name__ == '__main__':
    evaluate_vgg()

# def experiment(state, channel):
#     evaluate_vgg(state.learning_rate, dataset=state.dataset)

