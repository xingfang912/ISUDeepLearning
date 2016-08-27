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

from ISUDeepLearning.theano_ml.CNNs.C_CP_FC_SOFT import CNN as cnn_shallow
#from ISUDeepLearning.theano_ml.CNNs.C_CP_C_CP_FC_SOFT import CNN as cnn_deeper
#from ISUDeepLearning.theano_ml.CNNs.C_CPx3_FC_SOFT import CNN as cnn_deepest

import cPickle

if __name__ == '__main__':
    random_seed = 23455
    initial_learning_rate = 0.01
    learning_rate_decay = 0.998
    squared_filter_length_limit = 15.0
    n_epochs = 300
    batch_size = None
    img_cnls = 3
    nkerns = [32, 32]
    #feature map sizes = [224,222,110]
    mlp_layer_sizes = [ nkerns[-1]*110*110, 160, -1 ] # -1 = num classes. not known initially.
    # activation functions for the fully connected (FC) layer
    activations = [ Tanh ]
    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.99
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 100
    mom_params = {"start": mom_start, "end": mom_end, "interval": mom_epoch_interval}
    dropout = False
    if dropout:
        dropout_rates = [ 0.2, 0.5 ]
    else:
        dropout_rates = [ 0, 0 ]    
    use_bias = False
    results_file_name = None
    base_path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_'
    # ('face',[],base_path+'faces_1150_25s_46c_20_5_resized_pad_224'),
    # ('face_mirror',[],base_path+'faces_350_25s_46c_20_0_5_mid_dist_resized_pad_224'),
    datasets = [('eye_default',[],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_combined',['mirror','mirror','darken','darken','blur','blur','equalize_hist','equalize_hist','brighten','brighten','noise','noise'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_mirror',['mirror'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_brighten',['brighten'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_darken',['darken'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_eq_hist',['equalize_hist'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_blur',['blur'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_noise',['noise'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224'),
                ('eye_shift',['shift_up','shift_up','shift_down','shift_down','shift_left','shift_left','shift_right','shift_right'],base_path+'periocular_2300_50s_46c_30_20_resized_pad_224')]
    # below dataset has been corrected. Use it for your final data.
    # FaceVisibleStills_periocular_2300_50s_46c_30_20_resized_pad_224
    for (name, augmentation_schedule, dataset) in datasets:
        print 'training model:', name
        for k in range(1): # 5-fold cross-validation
            print 'k =', k
            num_classes, dataset_ = load_data(dataset, random_seed=random_seed)
            [(train_set_x, train_set_y),(valid_set_x, valid_set_y),(test_set_x, test_set_y)] = dataset_
            assert len(mlp_layer_sizes) - 1 == len(dropout_rates)
            mlp_layer_sizes[2] = num_classes # set num_classes based on dataset.
            batch_size = num_classes
            x = T.tensor4('x')  # the data is presented as rasterized images
            learning_rate = theano.shared(np.asarray(initial_learning_rate, dtype=theano.config.floatX))
            classifier = cnn_shallow(rng = np.random.RandomState(seed=random_seed),
                                 input = x.reshape((batch_size, img_cnls, 224, 224)),
                                 ninput_chnls = img_cnls,
                                 nkerns = nkerns,
                                 dropout_rates = dropout_rates,
                                 mlp_layer_sizes = mlp_layer_sizes,
                                 activations = activations,
                                 batch_size = batch_size,
                                 use_bias = use_bias)
            res = test_net(classifier = classifier,
                           num_classes = num_classes,
                           learning_rate = learning_rate,
                           learning_rate_decay = learning_rate_decay,
                           squared_filter_length_limit = squared_filter_length_limit,
                           n_epochs = n_epochs,
                           timeout = 300,
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
                           plot = False,
                           return_classifier = True,
                           augment_schedule = augmentation_schedule)
            #(best_model, hits, misses, roc, cmc, preds, lbls, plot_training, plot_test) = res
            with open(name+'_k'+str(k)+'_results.pickle', 'wb') as f:
                cPickle.dump(res, f, protocol=cPickle.HIGHEST_PROTOCOL)

