"""
Author: Ryan A. Dellana
Created: Dec. 15, 2015
Modified: Feb. 3, 2016

Description:
Formats a labeled image database for use in training convolutional NNs in Theano.
Output is pickled, gzipped, and written to disk in the current working directory.

Assumptions:
Current working directory is the top-level folder containing the class subfolders.
Each sub-folder contains samples of just one class.
Each sub-folder contains the same number of samples as the others.
There must be at least 3 samples for each class (preferably more than 10).
Image files must have an extension in the formats list.

Usage:
Suppose we want to train a convolutional NN to classify North Carolina tree species based on photos of their leaves.
Let there be 10 different species classes: Eastern Hemlock, Baldcypress, Black Willow, Swamp Cottonwood, Black Walnut, Pecan, 
River Birch, White Oak, Live Oak, American Elm.
1. Create a top-level directory. Lets name it NC_Forest_Trees.
2. Create subfolders for each of the species classes, Eastern_Hemlock, Baldcypress, Black_Willow, ...
3. Each subfolder should contain at least 10 different photos of each tree leaf (50+ is good).
   They can be of any resolution, dimension, image format, or colorspace, and they needn't be the same.
   The only constraint is that there be the same number of images per species/class folder.
4. Suppose you have this script in a folder one level above NC_Forest_Trees.
   Using the command line, change your working directory to NC_Forest_Trees (i.e.: cd NC_ForestTrees).
5. Now run this script using: python ../format_imgs_4_theano.py
   This script takes several optional parameters, which you will set depending on the topology of your Network.
   --width = should be the width/height of your input layer (defaults to 224).
   --color = 'gray' if your input layer has depth 1, 'bgr' if your input layer has depth 3 (defaults to bgr).
   --crop = if you set this to true, it'll crop the images to make them square, otherwise they're padded with zeros.
   --train, --val, --test = these specify how to partition the samples of each class into training, validation, and testing. 
     if you don't specify them, the data will be partitioned into 80% training, 10% validation, and 10% testing if there
     are fewer than 20 samples per class, otherwise, it'll partition it 50% 25% 25%.
   Another example: python ../format_imgs_4_theano.py --width=226 --color=bgr --crop=True --train=33 --val=16 --test=16

Bugs/Improvements:
Can use a lot of memory. May need to be modified for larger datasets.
Has only been tested on AT&T and Yale face datasets.
"""

import cv2
import numpy as np
import cPickle
import gzip
import os
import argparse

def resize_pad(img, target_width=224):
    """
    Convert an OpenCV image into a square with both width and height equal to target_width.
    Any extra space is filled via padding with zeros.
    Colorspace of img is preserved.

    :type target_width: int
    :param target_width: the height/width of your networks input layer.
    """
    w, h, chnls = 0, 0, 1
    if len(img.shape) == 3:
        (h, w, chnls) = img.shape
    elif len(img.shape) == 2:
        (h, w) = img.shape
    else:
        print 'Error: resize_pad: len(img.shape) must be 2 or 3'
        return None
    img_ = None
    if chnls == 1:
        img_ = np.zeros((target_width,target_width), np.uint8)
    else:
        img_ = np.zeros((target_width,target_width,chnls), np.uint8)
    if h == w:
        img_ = cv2.resize(img, (target_width, target_width))
    else:
        if h > w:
            new_width = int(w*(float(target_width)/h))
            img2 = cv2.resize(img, (new_width, target_width))
            margin = (target_width - new_width)/2
            img_[:,margin:margin+new_width] = img2[:,:]
        else: # w > h
            new_height = int(h*(float(target_width)/w))
            img2 = cv2.resize(img, (target_width, new_height))
            margin = (target_width - new_height)/2
            img_[margin:margin+new_height,:] = img2[:,:]
    return img_

def resize_crop(img, target_width=224):
    """
    Crop an OpenCV image into a square with both width and height equal to target_width.
    Colorspace of img is preserved.
    
    :type target_width: int
    :param target_width: the height/width of your networks input layer.
    """
    w, h, chnls = 0, 0, 1
    if len(img.shape) == 3:
        (h, w, chnls) = img.shape
    elif len(img.shape) == 2:
        (h, w) = img.shape
    else:
        print 'Error: resize_pad: len(img.shape) must be 2 or 3'
        return None
    img_ = None
    if h > w:
        margin = (h - w)/2
        img_ = img[margin:margin+w,:]
    else: # w > h:
        margin = (w - h)/2
        img_ = img[:,margin:margin+h]
    img_ = cv2.resize(img_, (target_width, target_width))
    return img_

def normalize_img(img, flatten=False):
    """
    Convert an OpenCV uint8 image to a float32 and normalize values to range of 0.0 - 1.0
    Option to flatten resulting array into a vector.    

    :type flatten: bool
    :param flatten: flatten the data instances, otherwise leave them as multi-dimensional arrays.
    """
    out_img = img.astype(dtype=np.float32)/256.0
    if flatten:
        out_img = out_img.flatten()
    return out_img

def format_img_4_theano(img, target_width=224, target_colorspace=None, crop=False, flatten=False):
    """
    Formats an image for use in training convolutional NNs in theano.
    
    :type target_width: int
    :param target_width: the height/width of your networks input layer.

    :type target_colorspace: str
    :param target_colorspace: convert images to gray or bgr colorspace.

    :type crop: bool
    :param crop: make images square via cropping, otherwise pad them with zeros.

    :type flatten: bool
    :param flatten: flatten the data instances, otherwise leave them as multi-dimensional arrays.
    """
    if target_colorspace is not None:
        if target_colorspace == 'gray' and len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif target_colorspace == 'bgr' and len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if crop:
        return normalize_img(resize_crop(img, target_width), flatten)
    else:
        return normalize_img(resize_pad(img, target_width), flatten)

def format_img_4_eigenfaces(img, target_width=256): # 128
    """
    Format an image for eigenfaces. Makes them black-and-white and square by padding.
    """
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return resize_pad(img, target_width)

def shuffle_in_unison(a, b):
    assert a.shape[0] == b.shape[0]
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(a.shape[0])
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def format_4_theano(target_width=224, target_colorspace=None, n_train=-1, n_val=-1, n_test=-1, crop=False, flatten=False):
    """
    Formats a labeled image database for use in training convolutional NNs in theano.
    Output is pickled, gzipped, and written to disk in the current working directory.

    Assumptions:
    > Current working directory is the top-level folder containing the class subfolders.
    > Each sub-folder contains samples of just one class.
    > Each sub-folder contains the same number of samples as the others.
    > There must be at least 3 samples for each class.
    > Image files must have an extension in the formats list.
    
    :type target_width: int
    :param target_width: the height/width of your networks input layer.

    :type target_colorspace: str
    :param target_colorspace: convert images to gray or bgr colorspace.

    :type n_train: int
    :param n_train: number of training instances per class.

    :type n_val: int
    :param n_val: number of validation instances per class.

    :type n_test: int
    :param n_test: number of test instances per class.

    :type crop: bool
    :param crop: make images square via cropping, otherwise pad them with zeros.

    :type flatten: bool
    :param flatten: flatten the data instances, otherwise leave them as multi-dimensional arrays.
    """
    formats = ['jpg','JPEG','png','gif','pgm','tiff','bmp','bad']
    class_folders = [ name for name in os.listdir('./') if os.path.isdir(os.path.join('./', name)) ]
    num_class_instances = -1
    for folder in class_folders:
        n = len([name for name in os.listdir('./'+folder) if (not os.path.isdir(os.path.join('./'+folder, name)) and name.split('.')[-1] in formats)])
        if num_class_instances == -1:
            num_class_instances = n
        elif n != num_class_instances:
            print 'Error: Subfolder '+folder+' does not contain the same number of instances as the others.'
            return
    assert num_class_instances >= 3
    assert n_train == -1 or n_train+n_val+n_test == num_class_instances
    if n_train == -1:
        if num_class_instances < 8:
            n_test = 1
        elif num_class_instances < 20: # at least 8 instances per class.
            n_test = int(num_class_instances*0.1)
        else: # if >= 20  then 50 25 25 split
            n_test = int(num_class_instances*0.25)
        n_val = n_test
        n_train = num_class_instances - n_test - n_val
    lbls, data = [], []
    for folder in class_folders:
        print 'processing class:', folder
        path_ = './'+folder+'/'
        #lbl = int(''.join([char for char in folder if char.isdigit()]))
        files = [name for name in os.listdir('./'+folder) if (not os.path.isdir(os.path.join('./'+folder, name)) and name.split('.')[-1] in formats)]
        #print 'processing files:', files
        class_samples = []
        for fname in files:
            img = cv2.imread(path_+fname)
            input_vector = format_img_4_theano(img, target_width, target_colorspace, crop, flatten)
            class_samples.append(input_vector)
        data.append(np.array(class_samples))
    for class_list in data:
        np.random.shuffle(class_list)
    lbls_train, lbls_val, lbls_test = [], [], []
    data_train, data_val, data_test = [], [], []
    for lbl, class_list in enumerate(data):
        for idx, item in enumerate(class_list):
            if idx+1 <= n_train: # training
                data_train.append(item)
                lbls_train.append(lbl)
            elif idx+1 <= n_train+n_val: # validation
                data_val.append(item)
                lbls_val.append(lbl)
            elif idx+1 <= n_train+n_val+n_test: # testing
                data_test.append(item)
                lbls_test.append(lbl)
    data_train, data_val, data_test = np.array(data_train), np.array(data_val), np.array(data_test)
    lbls_train, lbls_val, lbls_test = np.array(lbls_train), np.array(lbls_val), np.array(lbls_test)
    data_train, lbls_train = shuffle_in_unison(data_train, lbls_train)
    data_val, lbls_val = shuffle_in_unison(data_val, lbls_val)
    data_test, lbls_test = shuffle_in_unison(data_test, lbls_test)
    output = ((data_train,lbls_train), (data_val,lbls_val), (data_test,lbls_test))
    print 'n_train, n_val, n_test =', n_train, n_val, n_test
    print 'Serializing...'
    serialized = cPickle.dumps(output, protocol=2) # TODO warning! This may require a lot of memory! TODO
    print 'Gzipping...'
    crop_str = 'cropped' if crop else 'padded'
    color = 'bgr' if target_colorspace is None else target_colorspace
    output_name = os.getcwd().split('/')[-1]+'_4_theano_'+str(target_width)+'_'+str(np.max(lbls_val)+1)+'_'+str(n_train)+'_'+str(n_val)+'_'+str(n_test)+'_'+crop_str+'_'+color+'.pkl.gz'
    with gzip.open('./'+output_name, 'wb') as f:
        f.write(serialized)
    print 'created:', output_name

    # keep all class instances separate.
    # shuffle the instances within each class.
    # partition into training, validation, and test sets.
    # shuffle training, validation, and test sets.

def format_4_eigenfaces(target_width=256):
    """
    Formats a labeled image database for use in OpenCv Eigenfaces.
    Output is pickled and written to disk in the current working directory.

    Assumptions:
    > Current working directory is the top-level folder containing the class subfolders.
    > Each sub-folder contains samples of just one class.
    > Each sub-folder contains the same number of samples as the others.
    > There must be at least 3 samples for each class.
    > Image files must have an extension in the formats list.
    
    :type target_width: int
    :param target_width: the height/width of your networks input layer.
    """
    formats = ['jpg','JPEG','png','gif','pgm','tiff','bmp','bad']
    class_folders = [ name for name in os.listdir('./') if os.path.isdir(os.path.join('./', name)) ]
    num_class_instances = -1
    for folder in class_folders:
        n = len([name for name in os.listdir('./'+folder) if (not os.path.isdir(os.path.join('./'+folder, name)) and name.split('.')[-1] in formats)])
        if num_class_instances == -1:
            num_class_instances = n
        elif n != num_class_instances:
            print 'Error: Subfolder '+folder+' does not contain the same number of instances as the others.'
            return
    assert num_class_instances >= 3
    lbls, data = [], []
    for folder in class_folders:
        print 'processing class:', folder
        path_ = './'+folder+'/'
        lbl = int(''.join([char for char in folder if char.isdigit()]))
        files = [name for name in os.listdir('./'+folder) if (not os.path.isdir(os.path.join('./'+folder, name)) and name.split('.')[-1] in formats)]
        for fname in files:
            img = cv2.imread(path_+fname)
            input_vector = format_img_4_eigenfaces(img, target_width)
            data.append(input_vector)
            lbls.append(lbl)
    print 'Serializing...'
    output_name = os.getcwd().split('/')[-1]+'_4_eigen_'+str(target_width)+'.pkl'
    with open(output_name, 'wb') as f:
        serialized = cPickle.dump([data, lbls], f, protocol=2)
    print 'created:', output_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format Images for Theano')
    parser.add_argument('--mode', dest='mode', type=str, default='theano', choices=['theano', 'eigen'], required=False, help='mode: theano or eigenfaces.')
    parser.add_argument('--width', dest='width', type=int, default=224, required=False, help='width/height of target image')
    parser.add_argument('--color', dest='color', type=str, default='bgr', choices=['gray','bgr'], required=False, help='colorspace of target image (bgr, gray)')
    parser.add_argument('--crop', dest='crop', type=bool, default=False, required=False, help='crop images to square, otherwise pad.')
    parser.add_argument('--train', dest='n_train', type=int, default=-1, required=False, help='number of training samples per class.')
    parser.add_argument('--val', dest='n_val', type=int, default=-1, required=False, help='number of validation samples per class.')
    parser.add_argument('--test', dest='n_test', type=int, default=-1, required=False, help='number of testing samples per class.')
    args = parser.parse_args()
    if args.mode == 'theano':
        format_4_theano(args.width, args.color, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, crop=args.crop)
    elif args.mode == 'eigen':
        format_4_eigenfaces(args.width)
