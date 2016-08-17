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

import pickle

if __name__ == '__main__':
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
    mom_params = {"start": mom_start, "end": mom_end, "interval": mom_epoch_interval}
    dropout = False
    if dropout:
        dropout_rates = [ 0.2, 0.5 ] 
    else:
        dropout_rates = [ 0, 0 ]    
    use_bias = False
    results_file_name = None

    datasets = [('70s_5c', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_350_70s_5c_56_7_7_resized_pad_224'),
                ('10s_103c', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_1030_10s_103c_8_1_1_resized_pad_224'),
                ('60s_25c', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_1500_60s_25c_48_6_6_resized_pad_224'),
                ('40s_59c', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2360_40s_59c_32_4_4_resized_pad_224'),
                ('mirror', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224_augmented_mirror'),
                ('hist_equ', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224_augmented_hist_equ'),
                ('blur', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224_augmented_blur'),
                ('50s_46c', '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224')
    ]
    #dataset = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224'
    #dataset = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224_augmented'
    #dataset = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_faces_1150_25s_46c_20_2_3_resized_pad_224'

    for (name, dataset) in datasets:
        num_classes, dataset_ = load_data(dataset, random_seed=random_seed)
        assert len(mlp_layer_sizes) - 1 == len(dropout_rates)
        mlp_layer_sizes[2] = num_classes # set num_classes based on dataset.
        if batch_size is None:
            batch_size = num_classes
        n_epochs = int(2500.0/batch_size)
        x = T.tensor4('x')  # the data is presented as rasterized images
        learning_rate = theano.shared(np.asarray(initial_learning_rate, dtype=theano.config.floatX))
        classifier = cnn_shallow(rng = np.random.RandomState(seed=random_seed), 
                                 input = x.reshape((batch_size, 3, 224, 224)),
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
        print name
        print dataset
        print res
        with open(name+'.pickle', 'wb') as handle:
            pickle.dump(res, handle)
