"""
This tutorial introduces the tiny VGG architecture (C-CP-C-CP-C-CP-FC)
    Xing Fang (School of Information Technology, Illinois State University)
        and 
    Ryan Dellana (Department of Computer Science, North Carolina A&T State University)

Part of this program uses the code found at: https://github.com/mdenil/dropout

Date: May 27, 2016
"""
import numpy as np
import os, sys, copy
import time, math
import matplotlib.pyplot as plt
from collections import OrderedDict
import random, cv2

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.nnet import conv
import theano.printing
from theano.tensor.signal.pool import pool_2d

from ISUDeepLearning.DataMungingUtil import load_dataset

from ISUDeepLearning.DNN_functions import ReLU, Sigmoid, Tanh, HiddenLayer, dropout_from_layer, DropoutHiddenLayer

import cPickle


def load_data(path, colorspace='bgr', random_seed=None, nk=None, k=None):
    """path: path to dataset
       colorspace: specifies if image is color ('bgr') or grayscale ('gray')
       random_seed: seed used to shuffle images within each partition
       nk: if using k-fold cross validation this specifies k.
       k: specifies which k we're currently on.
       Note: if nk and k are specified, then the default partitions as specified by the folders are ignored.
    """
    num_classes, (train_set, valid_set, test_set) = load_dataset(path, colorspace, True, random_seed, nk, k)

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
    train_set_x, train_set_y = shared_dataset(train_set)
    if valid_set is None or len(valid_set) == 0:
        rval = [(train_set_x, train_set_y),(test_set_x, test_set_y)]
        return num_classes, rval
    else:
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        rval = [(train_set_x, train_set_y),(valid_set_x, valid_set_y),(test_set_x, test_set_y)]
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
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
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

# process raw minibatch test output and return cmc/roc data.
# cmc is a list where index [0-n] are accuracy for [rank-1 -> rank-n-1]
# roc is a dictionary where the keys are confidence thresholds from 0.01 to 1.0 with the item
# under each key being a tuple of the form (true_positive_rate, false_positive_rate)
def get_cmc_roc_data(softmax, classes):
    predictions = []
    labels = []
    for minibatch in softmax:
        for item in minibatch:
            predictions.append(item)
    for i in classes:
        labels.extend(i)
    # -------- cmc curve data ---------
    ranks = []
    for rank in range(1, 47):
        hits = 0
        for sample in range(len(predictions)):
            pred = predictions[sample]
            pred_ = zip(pred, range(len(pred)))
            pred_.sort(reverse=True)
            lbl = labels[sample]
            for i in range(0, rank):
                if pred_[i][1] == lbl:
                    hits += 1
        ranks.append(hits/float(len(predictions)))
    # -------- roc curve data ---------
    roc = {}
    n_samples = len(predictions)
    for i in range(1, 1001):
        detection_threshold = i/1000.0
        n_true_positives = 0
        n_false_positives = 0
        for sample in range(n_samples):
            pred = predictions[sample]
            pred_ = zip(pred, range(len(pred)))
            pred_.sort(reverse=True)
            lbl = labels[sample]
            prediction = pred_[0][1]
            confidence = pred_[0][0]
            if confidence >= detection_threshold:
                if prediction == lbl: # true positive
                    n_true_positives += 1
                else: # false positive
                    n_false_positives += 1
        roc[detection_threshold] = (n_true_positives/float(n_samples), n_false_positives/float(n_samples))
    return predictions, labels, ranks, roc

def plot_roc(roc):
    x = [roc[i/100.0][1] for i in range(1, 101)]
    y = [roc[i/100.0][0] for i in range(1, 101)]
    plt.plot(x, y)
    #plt.axis([0, 1.0, 0, 1.0])
    plt.title('ROC curve for CNN')
    plt.xlabel('False Accept Rate')
    plt.ylabel('True Accept Rate')
    plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.grid(True)
    plt.show()

def plot_cmc(ranks):
    plt.plot(range(1, len(ranks)+1), ranks[0:])
    plt.title('CMC Curve for CNN')
    plt.xlabel('Rank')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, len(ranks)+1))
    yticks = []
    rank_ = math.floor(ranks[0]*100.0)/100.0
    while rank_ < 1.01:
        yticks.append(rank_)
        rank_ += 0.01
    plt.yticks(yticks)
    plt.grid(True)
    plt.show()

def plot_training_error(plot_training, epoch_counter):
    Epoch = np.arange(1, epoch_counter+1)
    plt.plot(Epoch, plot_training)
    plt.grid(axis="y")
    plt.ylabel('Training Error',fontsize=14)
    plt.show()

def plot_testing_error(plot_test, epoch_counter):
    Epoch = np.arange(1, epoch_counter+1)
    plt.plot(Epoch, plot_test,color="r")
    plt.grid(axis="y")
    plt.xlabel('Iteration',fontsize=18)
    plt.ylabel('Testing Error',fontsize=14)
    plt.show()

brighten = None # global var that hold brightness image to be
                # added or subracted from samples.
random_noise = None # random noise image.

# augment a sample img using a specific operation.
# can handle color or gray-scale images.
# img should be a numpy array of dtype np.uint8
def augment(img, opp):
    if opp is None or opp == '':
        return img # no change.
    subopps = opp.split(' ')
    if 'mirror' in subopps:
        img = cv2.flip(img, flipCode=1)
    if 'brighten' in subopps or 'darken' in subopps:
        global brighten
        if brighten is None:
            brighten = np.zeros(shape=img.shape, dtype=np.uint8)
        if len(img.shape) == 3:
            brighten[:,:,2] = 80
        else:
            brighten[:,:] = 80
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if 'brighten' in subopps:
                hsv = cv2.add(hsv, brighten)
            else: # darken
                hsv = cv2.subtract(hsv, brighten)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else: # gray-scale
            if 'brighten' in subopps:
                img = cv2.add(img, brighten)
            else: # darken
                img = cv2.subtract(img, brighten)
    if 'equalize_hist' in subopps:
        if len(img.shape) == 3:
            y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
            y = cv2.equalizeHist(y)
            img = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCR_CB2BGR)
        else:
            img = cv2.equalizeHist(img)
    blur_opps = [opp for opp in subopps if 'blur' in opp]
    if len(blur_opps) > 0:
        blur = blur_opps[0]
        amount = 15
        if len(blur.split('_')) > 1:
            amount = int(blur.split('_')[1])
        assert amount % 2 == 1
        img = cv2.GaussianBlur(img,(amount,amount),0)
    if 'shift_' in opp:
        pass # crop a slightly smaller region out of the image flush with one of the sides.
        # percentage of image width. 0.9 and shift
        width = img.shape[0] # should be same as height
        new_width = int(width*0.95)
        gap = width-new_width
        start_px = gap/2
        if '_up' in opp:
            img = img[0:new_width,start_px:start_px+new_width]
        elif '_down' in opp:
            img = img[gap:,start_px:start_px+new_width]
        elif '_left' in opp:
            img = img[start_px:start_px+new_width,0:new_width]
        elif '_right' in opp:
            img = img[start_px:start_px+new_width,gap:]
        img = cv2.resize(img, (width, width))
    if 'noise' in opp:
        global random_noise
        if random_noise is None:
            random_noise = np.zeros(shape=img.shape, dtype=np.uint8)
            cv2.randu(random_noise, 0, 256)
            random_noise = cv2.GaussianBlur(random_noise,(5,5),0)
        img = cv2.addWeighted(img,0.90,random_noise,0.10,0)
    if 'glasses' in opp: # Periocular-specific
        pass # TODO TODO
    return img

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
        timeout,
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
        plot = False,
        return_classifier = False,
        augment_schedule = []):

    #[(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)] = dataset
    [(train_set_x, train_set_y),(valid_set_x, valid_set_y),(test_set_x, test_set_y)] = dataset
    
    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
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

    softmax_predictions = theano.function(inputs=[index],
            outputs=classifier.p_y_given_x_(),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]},
            on_unused_input='ignore')

    test_labels = theano.function(inputs=[index],
            outputs=test_set_y[index * batch_size:(index + 1) * batch_size])

    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    #validate_model = theano.function(inputs=[index],
    #        outputs=classifier.errors(y),
    #        givens={
    #            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
    #            y: valid_set_y[index * batch_size:(index + 1) * batch_size]},
    #        on_unused_input='ignore')
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

    best_test_error = np.inf
    best_test_score = np.inf
    best_iter_test = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

    plot_training = []
    plot_test = []
    best_model = None # saves the best model to be returned by function.
    #layer0_weights = None

    train_set_x_backup = train_set_x.get_value()
    while epoch_counter < n_epochs:
        # Train this epoch
        # augment the images
        if len(augment_schedule) > 0:
            opp = augment_schedule[epoch_counter % len(augment_schedule)]
            train_set_x_augment = copy.deepcopy(train_set_x_backup)
            print 'augmenting epoch with operation:', opp
            for i in range(len(train_set_x_augment)):
                # augment even images on even epochs and odd images on odd epochs.
                if (epoch_counter % 2 == 0 and i % 2 == 0) or (epoch_counter % 2 == 1 and i % 2 == 1): 
                    img = train_set_x_augment[i]
                    img = (img*256.0).astype(dtype=np.uint8)
                    if img.shape[2] == 1: # if it's a one channel image
                        img = np.reshape(img, (img.shape[0], img.shape[1]))
                    img = augment(img, opp)
                    img = img.astype(dtype=np.float32)/256.0
                    if len(img.shape) == 2:
                        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                    train_set_x_augment[i] = img
            train_set_x.set_value(train_set_x_augment)
        epoch_counter = epoch_counter + 1
        minibatch_avg_cost = 0
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost += train_model(epoch_counter, minibatch_index)
        plot_training.append(minibatch_avg_cost/n_train_batches)
        test_losses = [test_model(i) for i in xrange(n_test_batches)]
        this_test_error = np.mean(test_losses)
        plot_test.append(this_test_error)
        print "epoch {}, test error {}%, train error {}, learning_rate={}{}".format(
               epoch_counter, this_test_error*100.0, plot_training[-1],
               learning_rate.get_value(borrow=True),
               " **" if this_test_error < best_test_error else ""
               )
        #print 'predictions', test_softmax_predictions
        if this_test_error < best_test_error:
            best_test_error = this_test_error
            best_iter_test = epoch_counter
            test_softmax_predictions = [softmax_predictions(i) for i in xrange(n_test_batches)]
            test_labels_ = [test_labels(i) for i in xrange(n_test_batches)]
            #best_model_ = [param.get_value() for param in classifier.params]
            #best_model = cPickle.dumps(best_model_, protocol=cPickle.HIGHEST_PROTOCOL) # doesn't work TODO TODO
            if return_classifier:
                best_model = cPickle.dumps(classifier.params, protocol=cPickle.HIGHEST_PROTOCOL)
            # TODO extract filter images.
            #layer0_weights = classifier.layer0.W.get_value()
        if (timeout is not None) and (epoch_counter - best_iter_test >= timeout):
            break
        if decay:
            new_learning_rate = decay_learning_rate()
    end_time = time.clock()
    print >> sys.stderr, (('Optimization complete. Best test score of %f %% '
           'obtained at epoch %i') %
           (best_test_error * 100., best_iter_test))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    preds, lbls, cmc, roc = get_cmc_roc_data(test_softmax_predictions, test_labels_)
    test_set = test_set_x.get_value()
    misses = []
    hits = []
    for sample in range(len(preds)):
        pred = preds[sample]
        pred_ = zip(pred, range(len(pred)))
        pred_.sort(reverse=True)
        prediction = pred_[0][1]
        lbl = lbls[sample]
        if prediction != lbl:
            misses.append(test_set[sample])
        else:
            hits.append(test_set[sample])
    #for idx, img in enumerate(d):
    #    img_ = (img*256.0).astype(dtype=np.uint8)
    #    cv2.imwrite('missed_img'+str(idx)+'.jpg', img_)
    if plot:
        plot_cmc(cmc)
        plot_roc(roc)
        plot_training_error(plot_training, epoch_counter)
        plot_testing_error(plot_test, epoch_counter)
    return (best_model, hits, misses, roc, cmc, preds, lbls, plot_training, plot_test)
