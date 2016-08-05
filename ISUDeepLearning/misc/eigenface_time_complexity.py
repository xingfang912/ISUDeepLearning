import cv2
from ISUDeepLearning.DataMungingUtil import load_dataset
import time
import numpy as np

#path = '/home/ryan/datasets/CroppedYale_resized_pad_224'
#path = '/home/ryan/datasets/iPhone5_v2_resized_pad_224'
#path = '/home/ryan/datasets/SamsungGalaxyS4_resized_pad_224'
#path = '/home/ryan/datasets/iPhone5_plus_SamsungGalaxyS4_pad_224'
path = '/home/ryan/datasets/iPhone5_SamsungGalaxyS4_SamsungGalaxyTab2_resize_pad_224'
#path = '/home/ryan/datasets/Cropped_Yale_Subset_Periocular_resized_pad_224'

num_classes, ((train_x, train_y), (val_x, val_y), (test_x, test_y)) = load_dataset(path, colorspace='gray', normalize=False)

for i in [2, 4, 8, 16, 32, 64, 128, 256]:
    avg = 0.0
    for x in range(0, 4):
        eigen = cv2.createEigenFaceRecognizer()
        start = time.time()
        eigen.train(train_x[0:i], train_y[0:i])
        avg += time.time() - start
    print i, avg/4.0
