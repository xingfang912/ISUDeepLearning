"""
Author: Ryan A. Dellana
Created: Dec. 15, 2015
Modified: June. 21, 2016
"""

import cv2
# from skimage.feature import local_binary_pattern
import numpy as np
import os, shutil, argparse, random
import lmdb
import caffe

# Does a parallel shuffle of two numpy arrays.
# Note: Shuffle is not in-place. New arrays are returned.
def shuffle_in_unison(a, b, random_seed=43):
    assert a.shape[0] == b.shape[0]
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    rng = np.random.RandomState()
    if random_seed is not None:
        rng = np.random.RandomState(seed=random_seed)
        print 'init shuffle_in_unison with random seed', random_seed
    else:
        print 'init shuffle_in_unison random seed not specified'
    permutation = rng.permutation(a.shape[0])
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

# Reads in an image from 'path'.
# 'colorspace' parameter determines if image is read in as 'bgr' or 'gray'.
# if 'normalize' is True, then returns as float32 array with (0->255) mapped to (0->1.0) 
#    otherwise its the standard numpy.uint8 format used by opencv.
def read_img(path, colorspace='bgr', normalize=True):
    img = cv2.imread(path)
    if colorspace == 'gray' and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif colorspace == 'bgr' and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if normalize:
        img = img.astype(dtype=np.float32)/256.0
    return img

# Read in a dataset from "path", which is assumed to contain a "training", "validation", and "testing" folder.
# 'colorspace' parameter determines if image is read in as 'bgr' or 'gray'.
# if 'normalize' is True, then returns as float32 array with (0->255) mapped to (0->1.0) 
#    otherwise its the standard numpy.uint8 format used by opencv.
# 'random_seed' is used to shuffle the data after it's read in, so that order in folders doesn't matter.
def load_dataset(path, colorspace='bgr', normalize=True, random_seed=43):
    path_train = os.path.join(path, 'training')
    path_val = os.path.join(path, 'validation')
    path_test = os.path.join(path, 'testing')
    assert os.path.exists(path_train)
    assert os.path.exists(path_val)
    assert os.path.exists(path_test)
    class_folders = _subdirectories(path_train)
    data_train, data_val, data_test = [], [], []
    lbls_train, lbls_val, lbls_test = [], [], []
    num_classes = len(class_folders)
    for lbl, folder in enumerate(class_folders):
        path_train_ = os.path.join(path_train, folder)
        path_val_ = os.path.join(path_val, folder)
        path_test_ = os.path.join(path_test, folder)
        train_files = _img_files(path_train_)
        val_files = _img_files(path_val_)
        test_files = _img_files(path_test_)
        for f in train_files:
            data_train.append(read_img(os.path.join(path_train_, f), colorspace, normalize))
            lbls_train.append(lbl)
        for f in val_files:
            data_val.append(read_img(os.path.join(path_val_, f), colorspace, normalize))
            lbls_val.append(lbl)
        for f in test_files:
            data_test.append(read_img(os.path.join(path_test_, f), colorspace, normalize))
            lbls_test.append(lbl)
    data_train, data_val, data_test = np.array(data_train), np.array(data_val), np.array(data_test)
    lbls_train, lbls_val, lbls_test = np.array(lbls_train), np.array(lbls_val), np.array(lbls_test)
    data_train, lbls_train = shuffle_in_unison(data_train, lbls_train, random_seed)
    data_val, lbls_val = shuffle_in_unison(data_val, lbls_val, random_seed)
    data_test, lbls_test = shuffle_in_unison(data_test, lbls_test, random_seed)
    train_set, valid_set, test_set = ((data_train,lbls_train), (data_val,lbls_val), (data_test,lbls_test))
    return num_classes, (train_set, valid_set, test_set)

formats = ['jpg','JPEG','png','gif','pgm','tiff','bmp','bad']

# If all samples are in the same directory, this function 
# will create class folders and move samples into them accordingly.
# It is assumed that the class is specified at the beginning the filename of each sample and separated
# from the remainder of the filename using one of the following delimiters (' ', '_', '-', '.')
# Note that this function DOES NOT COPY the files, it just moves them.
def group_into_class_folders(path, delimiters=[' ', '_', '-', '.']):
    assert (os.path.exists(path) and os.path.isdir(path))
    files = _img_files(path)
    class_folders = {}
    for f in files:
        f_ = f.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        f_ = f_.split(' ')[0]
        if f_ in class_folders:
            class_folders[f_].append(f)
        else:
            class_folders[f_] = [f]
    for k in class_folders:
        os.mkdir(path+'/'+k)
        for f in class_folders[k]:
            shutil.move(path+'/'+f, path+'/'+k+'/'+f)

# Forces all classes to have "n" samples.
# If a given class has fewer than "n" samples, then it's class directory is deleted.
# If it has more than "n" samples, then some are deleted to make it equal to "n"
# IMPORTANT: MODIFIES IN PLACE. DELETION IS PERMANENT.
def equalize_num_samples(path, n):
    assert (os.path.exists(path) and os.path.isdir(path))
    assert (type(n) is int) and (n > 0)
    class_folders = _subdirectories(path)
    for folder in class_folders:
        path_ = os.path.join(path, folder)
        pictures = _img_files(path_)
        n_ = len(pictures)
        if n_ < n:
            shutil.rmtree(path_) # delete the directory.
        elif n_ > n:
            num_remove = n_ - n # delete some samples from the end.
            remove_list = pictures[0:num_remove]
            for pic in remove_list:
                os.remove(path_+'/'+pic)

# Partitions samples into training, testing, and validation folders 
# with the number of samples in each specified by the 'training', 'validation', 
# and 'testing' parameters.
# Assumes each class has its own folder within 'path'.
# If run on previously partitioned data, it'll unpartition before repartitioning.
# DOES NOT MAKE COPY. MODIFIES IN PLACE.
def partition(path, training, validation, testing, seed=43):
    print 'partitioning...'
    assert (os.path.exists(path) and os.path.isdir(path))
    assert (type(training) is int) and (training > 0)
    assert (type(validation) is int) and (validation > 0)
    assert (type(testing) is int) and (testing >= 0)
    class_folders = _subdirectories(path)
    if len(class_folders) <= 3 and 'training' in class_folders: # need to unpartition.
        unpartition(path)
    class_folders = _subdirectories(path)
    path_train = os.path.join(path, 'training')
    path_val = os.path.join(path, 'validation')
    path_test = os.path.join(path, 'testing')
    os.mkdir(path_train)
    for c in class_folders:
        os.mkdir(os.path.join(path_train, c))
    shutil.copytree(path_train, path_val)
    shutil.copytree(path_train, path_test)
    n = training + validation + testing
    seq = range(0, n)
    random.seed(seed)
    random.shuffle(seq)
    print 'Shuffle:', seq
    for c in class_folders:
        class_path = os.path.join(path, c)
        img_files = _img_files(class_path)
        if c == '001':
            print 'class_path =', class_path
            print 'img_files =', img_files
        for i in range(0, training):
            f = os.path.join(class_path, img_files[seq[i]])
            if c == '001':
                print f
            shutil.move(f, os.path.join(path_train, c))
        for i in range(training, training+validation):
            f = os.path.join(class_path, img_files[seq[i]])
            if c == '001':
                print f
            shutil.move(f, os.path.join(path_val, c))
        for i in range(training+validation, training+validation+testing):
            f = os.path.join(class_path, img_files[seq[i]])
            if c == '001':
                print f
            shutil.move(f, os.path.join(path_test, c))
        shutil.rmtree(class_path)

# Reverses the partition operation.
# Specifically, it gets rid of the 'training', 'validation', and 'testing' folders and merges
# their subfolders in 'path'.
def unpartition(path):
    print 'unpartitioning...'
    assert (os.path.exists(path) and os.path.isdir(path))
    path_train = os.path.join(path, 'training')
    path_val = os.path.join(path, 'validation')
    path_test = os.path.join(path, 'testing')
    assert os.path.exists(path_train)
    assert os.path.exists(path_val)
    assert os.path.exists(path_test)
    class_folders = _subdirectories(path_train)
    class_folders_v = _subdirectories(path_val)
    class_folders_t = _subdirectories(path_test)
    assert (class_folders == class_folders_v) and (class_folders == class_folders_t)
    for c in class_folders:
        shutil.move(os.path.join(path_train, c), path) # move to root folder.
    for c in class_folders:
        files = _img_files(os.path.join(path_val, c))
        for fname in files:
            shutil.move(os.path.join(path_val, c+'/'+fname), os.path.join(path, c))
        files = _img_files(os.path.join(path_test, c))
        for fname in files:
            shutil.move(os.path.join(path_test, c+'/'+fname), os.path.join(path, c))
    shutil.rmtree(path_train)
    shutil.rmtree(path_val)
    shutil.rmtree(path_test)

# get names of image files in path.
def _img_files(path):
    return [n for n in os.listdir(path) if (not os.path.isdir(os.path.join(path, n)) and n.split('.')[-1] in formats)]

# get names of subdirectories in path.
def _subdirectories(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

# makes a copy of images and directories.
# New images are result of local binary pattern with histogram equalization.
"""
def filter_lbp(path):
    _resize(path, mode='lbp')

def _lbp(img):
    radius = 3
    no_points = 8 * radius
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ = local_binary_pattern(img_, no_points, radius, method='uniform')
    img_ = cv2.equalizeHist(img_.astype(np.uint8))
    img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR)
    return img_
"""

# Performs resize operation with padding, see documentation for '_resize'
def resize_pad(path, target_width=224):
    _resize(path, target_width, 'pad')

# Performs resize operation with cropping, see documentation for '_resize'
def resize_crop(path, target_width=224):
    _resize(path, target_width, 'crop')

# Resizes all images found recursively within 'path' and saves them to a new directory
# up one level from 'path' while also copying the directory structure within 'path'
# All images are resized to be squares with a width and height equal to 'target_width'
# Images are either padded or cropped depending on the 'mode' parameter ('pad', 'crop')
def _resize(path, target_width=224, mode='pad'):
    assert (os.path.exists(path) and os.path.isdir(path))
    assert (type(target_width) is int) and (target_width > 0)
    up_one = '/'+'/'.join(path.strip('/').split('/')[0:-1])
    target_path_name = path.strip('/').split('/')[-1]+'_resized_'+mode+'_'+str(target_width)
    target_path =  os.path.join(up_one, target_path_name)
    os.mkdir(target_path)
    for root, dirs, files in os.walk(path, topdown=True):
        relpath = os.path.relpath(root, path)
        newpath = up_one+'/'+target_path_name+'/'+relpath
        print newpath
        for name in dirs:
            os.mkdir(os.path.join(newpath, name))
        for name in files:
            f = os.path.join(root, name)
            img_orig = cv2.imread(f)
            if img_orig is not None:
                img = None
                if mode == 'pad':
                    img = _resize_pad(cv2.imread(f), target_width)
                elif mode == 'crop':
                    img = _resize_crop(cv2.imread(f), target_width)
                #elif mode == 'lbp':
                #    img = _lbp(cv2.imread(f))
                f2 = os.path.join(newpath, '.'.join(name.split('.')[0:-1])+".bmp")
                print 'f2 = ', f2
                cv2.imwrite(f2, img)

# Resizes a numpy.uint8 'img' to a square of 'target_width'. Makes it square via padding.
def _resize_pad(img, target_width=224):
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

# Resizes a numpy.uint8 'img' to a square of 'target_width'. Makes it square via cropping.
def _resize_crop(img, target_width=224):
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


# ======================================================================================

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
        return normalize_img(_resize_crop(img, target_width), flatten)
    else:
        return normalize_img(_resize_pad(img, target_width), flatten)

def format_img_4_caffe(img, target_width=224, target_colorspace=None, crop=False):
    if target_colorspace is not None:
        if target_colorspace == 'gray' and len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif target_colorspace == 'bgr' and len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    output_img = None
    if crop:
        output_img = _resize_crop(img, target_width)
    else:
        output_img = _resize_pad(img, target_width)
    return output_img

# Format an image for eigenfaces. Makes them black-and-white and square by padding.
def format_img_4_eigenfaces(img, target_width=256): # 128
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _resize_pad(img, target_width)

# requires images to be in class folders 
def format_4_caffe(path, target_width=224, target_colorspace=None, crop=False):
    assert (os.path.exists(path) and os.path.isdir(path))
    formats = ['jpg','JPEG','png','gif','pgm','tiff','bmp','bad']
    class_folders = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    # --------- Determine how many samples there are of each class -------------------
    num_class_instances = -1
    for folder in class_folders:
        contents = os.listdir(os.path.join(path, folder))
        n = len([name for name in contents if (not os.path.isdir(os.path.join(os.path.join(path, folder), name)) and name.split('.')[-1] in formats)])
        if num_class_instances == -1:
            num_class_instances = n
        elif n != num_class_instances:
            print 'Error: Subfolder '+folder+' does not contain the same number of instances as the others.'
            return
    assert num_class_instances >= 3
    # ---------------------------------------------------------------------------------
    lbls, imgs = [], []
    for lbl, folder in enumerate(class_folders):
        print 'processing class:', folder
        path_ = './'+folder+'/'
        # lbl = int(''.join([char for char in folder if char.isdigit()]))
        files = [name for name in os.listdir('./'+folder) if (not os.path.isdir(os.path.join('./'+folder, name)) and name.split('.')[-1] in formats)]
        for fname in files:
            img = cv2.imread(path_+fname)
            img = format_img_4_caffe(img, target_width=target_width, target_colorspace=target_colorspace, crop=crop)
            imgs.append(img)
            lbls.append(lbl)
    N = len(imgs)
    map_size = N*imgs[0].nbytes * 10
    env = lmdb.open('lmdb', map_size=map_size)
    with env.begin(write=True) as txn:
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            height, width, channels = imgs[i].shape
            datum.channels = channels
            datum.height = height
            datum.width = width
            datum.data = imgs[i].tobytes()
            datum.label = int(lbls[i])
            # The encode is only essential in Python 3
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
    print 'created: lmdb'


# =============== DEPRECATED METHODS ================== TODO

"""
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
"""

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
