# from upper-left corner:
# (0, 0) to (75, 75) (row, col)
# from upper-right corner:
# (0, 93) to (75, 168)

import DataMungingUtil as dm
import cv2

path = '/home/ryan/datasets/CroppedYale_Subset'
path_dest = '/home/ryan/datasets/Cropped_Yale_Subset_Periocular'

images = []

dirs = dm._subdirectories(path)
for d in dirs:
  path_ = path+'/'+d
  imgs = dm._img_files(path_)
  for img in imgs:
    path__ = path_+'/'+img
    print path__
    img_ = cv2.imread(path__)
    left_eye = img_[0:75, 0:75, :]
    cv2.imwrite(path_dest+'/'+img, left_eye)

"""
/home/ryan/datasets/Cropped_Yale_Subset_Periocular

LBPH:
training took 13.9026808739 seconds
mixed up 19 and 35
mixed up 19 and 16
mixed up 17 and 7
mixed up 19 and 26
mixed up 19 and 15
mixed up 17 and 34
mixed up 19 and 21
mixed up 19 and 6
mixed up 19 and 29
mixed up 19 and 22
mixed up 19 and 21
mixed up 19 and 12
mixed up 19 and 10
mixed up 19 and 31
mixed up 19 and 7
mixed up 19 and 8
mixed up 19 and 15
mixed up 19 and 22
mixed up 19 and 21
mixed up 19 and 21
mixed up 19 and 14
mixed up 19 and 32
mixed up 19 and 20
mixed up 19 and 29
mixed up 19 and 23
mixed up 17 and 38
mixed up 19 and 1
mixed up 17 and 37
mixed up 19 and 4
mixed up 31 and 6
mixed up 19 and 14
mixed up 17 and 34
mixed up 17 and 2
mixed up 19 and 25
mixed up 19 and 24
mixed up 19 and 9
mixed up 19 and 27
mixed up 19 and 11
mixed up 19 and 17
mixed up 17 and 35
mixed up 19 and 37
mixed up 17 and 29
mixed up 19 and 3
mixed up 19 and 28
mixed up 19 and 24
mixed up 19 and 38
mixed up 19 and 18
mixed up 19 and 18
mixed up 19 and 6
mixed up 19 and 3
mixed up 19 and 3
mixed up 17 and 31
mixed up 19 and 11
mixed up 19 and 13
mixed up 19 and 4
mixed up 19 and 27
mixed up 19 and 3
mixed up 19 and 9
mixed up 19 and 15
mixed up 19 and 20
mixed up 19 and 37
mixed up 17 and 32
mixed up 19 and 24
mixed up 19 and 30
mixed up 19 and 38
mixed up 19 and 14
mixed up 17 and 1
mixed up 19 and 8
mixed up 19 and 24
mixed up 19 and 3
mixed up 19 and 6
mixed up 19 and 16
mixed up 17 and 13
mixed up 19 and 9
mixed up 19 and 2
mixed up 17 and 23
mixed up 19 and 26
error rate: 40.53%

Eigen:
training took 588.177182913 seconds
mixed up 13 and 3
mixed up 5 and 9
mixed up 1 and 18
mixed up 11 and 7
mixed up 1 and 21
mixed up 28 and 9
mixed up 27 and 2
mixed up 16 and 35
mixed up 37 and 36
mixed up 29 and 6
mixed up 36 and 1
mixed up 33 and 23
mixed up 15 and 30
mixed up 6 and 26
mixed up 6 and 11
mixed up 33 and 3
mixed up 37 and 16
mixed up 19 and 31
mixed up 16 and 37
mixed up 36 and 3
mixed up 29 and 13
mixed up 33 and 38
mixed up 16 and 21
mixed up 1 and 8
mixed up 7 and 2
mixed up 12 and 21
mixed up 17 and 10
mixed up 12 and 1
mixed up 6 and 29
mixed up 36 and 28
mixed up 9 and 4
mixed up 6 and 13
mixed up 16 and 37
mixed up 33 and 16
mixed up 9 and 6
mixed up 26 and 37
mixed up 17 and 34
mixed up 2 and 35
mixed up 25 and 29
mixed up 16 and 11
mixed up 20 and 7
mixed up 25 and 6
error rate: 22.11%

CNN:
training @ iter =  1519
0.0633819326758
epoch 40, minibatch 38/38, validation error 17.368421 %
Optimization complete.
Best validation score of 15.263158 % obtained at iteration 1216, with test performance 14.912281 %
The code for file Tiny_VGG_v2.py ran for 36.45m

/home/ryan/datasets/iPhone5_SamsungGalaxyS4_SamsungGalaxyTab2_resize_pad_224
LBPH:
training took 24.4896230698 seconds
mixed up 28 and 68
mixed up 10 and 31
mixed up 59 and 38
mixed up 72 and 40
mixed up 66 and 55
mixed up 15 and 2
mixed up 46 and 69
mixed up 41 and 50
mixed up 13 and 9
mixed up 60 and 51
mixed up 26 and 61
error rate: 3.77%

Eigen:
training took 1623.00037289 seconds
mixed up 39 and 69
mixed up 50 and 31
mixed up 66 and 55
mixed up 45 and 70
mixed up 56 and 50
mixed up 45 and 68
mixed up 30 and 38
mixed up 15 and 2
mixed up 15 and 2
mixed up 52 and 9
mixed up 54 and 73
mixed up 26 and 61
mixed up 27 and 51
mixed up 30 and 40
mixed up 56 and 17
error rate: 5.14%

CNN:
training @ iter =  1399
0.0193407796323
epoch 40, minibatch 35/35, validation error 7.876712 %
Optimization complete.
Best validation score of 7.876712 % obtained at iteration 1260, with test performance 8.219178 %
The code for file Tiny_VGG_v2.py ran for 63.37m

================

The Relative Performance of Convolutional Neural Networks on Periocular Biometrics

The periocular region is the area around the eye, including the eye-lid and brow, that contains a relatively large amount of stable patterns useful in biometric authentication. When performing continuous authentication using the standard cameras found in smart phones and tablets, it's typical for only a portion of the face to be visible at any given time. To overcome the shortcomings of traditional face recognition approaches in this context, one approach is to perform periocular region detection and classification. Convolutional Neural Networks have become the standard in image classification, with one particular topology, VGG-Net, performing quite well relative to it's complexity. As such, we measure the performance of this CNN on periocular classificiation for two different datasets relative to two well-studied algorithms, Eigen face, and Local Binary Pattern Histograms (LBPH). The BIPLAB MICHE dataset consists of 40 non-ideal periocular images from each of 75 subjects across three different devices (iPhone5, Samsung Galaxy S4, Samsung Galaxy Tablet 2). MICHE is specifically useful for comparing images of the same subject taken from cameras with different optical properties including aspect ratio. Our second dataset was formed from the periocular regions of a subset of the cropped Yale face dataset, which focuses on the effects of lighting and shadows (38 subjects with 46 samples each). All images were reformatted to a size of 224x224 while keeping the aspect ratio constant using padding. MICHE was partitioned into 35 training, 4 validation, and 1 test sample per class, while Yale was given a 38-5-3 split. All Three classifiers were compared against the same datasets with the same partitioning. For Eigen face and LBPH we used the open-source implementation found in the OpenCV face recognition library. Our VGG-Net implementation, "Tiny-VGG" was built using the Theano framework and trained on a Quadro K6000 GPU. On the BIPLAB MICHE dataset, LBPH, Eigen face, and CNN performed with 96.23%, 94.86%, and 92.13% accuracy respectively. On the Yale Periocular dataset they reversed ranking with 59.47%, 77.89%, and 84.74%. We observed that, on realistic datasets such as MICHE, LBPH is the best overall value in terms of accuracy to training time ratio and has the distinct advantage of being continuously trainable. However, it performed very poorly on the Yale dataset which contains harsh shadows that spoof the local binary patterns. Eigen and CNN were good all-around performers, with Eigen outperforming CNN by 2.73% on MICHE while CNN did 6.85% better on Yale. However, both must be retrained with the addition of any new classes, wich takes a long time with CNN requiring a relatively powerful GPU. Work continues on data augmentation methods for the CNN to avoid over-fitting. These will be specific to periocular data and may use another CNN to find the eye center as a reference for image purtubations. Another avenue of exploration is CNN pre-training on a general periocular dataset.

CNN: avg = 88.435%
Eigen: avg = 86.375%
LBPH: avg = 77.85%

"""

