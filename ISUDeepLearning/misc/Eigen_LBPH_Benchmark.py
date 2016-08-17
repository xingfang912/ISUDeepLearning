import cv2
from ISUDeepLearning.DataMungingUtil import load_dataset
import time
import numpy as np


def test_classifier(classifier, val_x, val_y, test_x, test_y):
    miss_hist = {}
    hit, miss = 0, 0
    for i in range(0, len(val_x)):
        cls, conf = classifier.predict(val_x[i])
        #cls, conf = eigen.predict(val_x[i])
        #print cls
        if cls == val_y[i]:
            hit += 1
        else:
            miss += 1
            missed = cls+1
            missed2 = val_y[i]+1
            if missed in miss_hist:
                miss_hist[missed] += 1
            else:
                miss_hist[missed] = 1
            if missed2 in miss_hist:
                miss_hist[missed2] += 1
            else:
                miss_hist[missed2] = 1
            #print 'mixed up', cls+1, 'and', val_y[i]+1
    test_hit = 0
    test_miss = 0
    for i in range(0, len(test_x)):
        cls, conf = classifier.predict(test_x[i])
        if cls == test_y[i]:
            test_hit += 1
        else:
            test_miss += 1
    return miss_hist, hit, miss, test_hit, test_miss

#path = '/home/ryan/datasets/CroppedYale_resized_pad_224'
#path = '/home/ryan/datasets/iPhone5_v2_resized_pad_224'
#path = '/home/ryan/datasets/SamsungGalaxyS4_resized_pad_224'
#path = '/home/ryan/datasets/iPhone5_plus_SamsungGalaxyS4_pad_224'
#path = '/home/ryan/datasets/iPhone5_SamsungGalaxyS4_SamsungGalaxyTab2_resize_pad_224'
#path = '/home/ryan/datasets/Cropped_Yale_Subset_Periocular_resized_pad_224'
#path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_40s_51c_32t_4v_4tst_resized_pad_224'
#path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_50s_27c_40t_5v_5tst_resized_pad_224'
#path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_10s_98c_8t_1v_1tst_resized_pad_224'
#path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_20s_78c_16t_2v_2tst_resized_pad_224'
#path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224'
path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_periocular_2300_50s_46c_40_5_5_resized_pad_224_augmented'
#path = '/home/ryan/datasets/MBGC_datasets/V2/MBGC-V2-data/FaceVisibleStills_faces_1150_25s_46c_20_2_3_resized_pad_224'

num_classes, ((train_x, train_y), (val_x, val_y), (test_x, test_y)) = load_dataset(path, colorspace='gray', normalize=False)

eigen = cv2.createEigenFaceRecognizer()
lbph = cv2.createLBPHFaceRecognizer()

start = time.time()
#eigen.train(train_x, train_y)
lbph.train(train_x, train_y)
print 'training took', time.time() - start, 'seconds'
miss_hist, hit, miss, test_hit, test_miss = test_classifier(lbph, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y)
print 'lbph validation error'
print 'error rate:', str(round((float(miss)/float(hit+miss))*100.0, 2))+'%'
print 'lbph testing error'
print 'error rate:', str(round((float(test_miss)/float(test_hit+test_miss))*100.0, 2))+'%'
print ''
start = time.time()
eigen.train(train_x, train_y)
print 'eigen training took', time.time() - start, 'seconds'
miss_hist, hit, miss, test_hit, test_miss = test_classifier(eigen, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y)
print 'eigen validation error'
print 'error rate:', str(round((float(miss)/float(hit+miss))*100.0, 2))+'%'
print 'eigen testing error'
print 'error rate:', str(round((float(test_miss)/float(test_hit+test_miss))*100.0, 2))+'%'
print ''



"""
confusing_classes = []
sum = 0
for k in miss_hist:
    sum += miss_hist[k]
avg = sum/len(miss_hist.keys())
for k in miss_hist:
    if miss_hist[k] > avg*2:
        confusing_classes.append(k)
print 'confusing classes:'
print confusing_classes

train_x2 = []
train_y2 = []
for i in range(0, len(train_y)):
    if train_y[i] in confusing_classes:
        train_x2.append(train_x[i])
        train_y2.append(train_y[i])
train_y2 = np.array(train_y2)

eigen.train(train_x2, train_y2)

hit, miss = 0, 0
for i in range(0, len(val_x)):
    cls, conf = lbph.predict(val_x[i])
    if cls in confusing_classes:
        cls, conf = eigen.predict(val_x[i])
    #print cls
    if cls == val_y[i]:
        hit += 1
    else:
        miss += 1
        missed = cls+1
        missed2 = val_y[i]+1
        if missed in miss_hist:
            miss_hist[missed] += 1
        else:
            miss_hist[missed] = 1
        if missed2 in miss_hist:
            miss_hist[missed2] += 1
        else:
            miss_hist[missed2] = 1
        #print 'mixed up', cls+1, 'and', val_y[i]+1

test_hit = 0
test_miss = 0
for i in range(0, len(test_x)):
    cls, conf = lbph.predict(test_x[i])
    if cls in confusing_classes:
        cls, conf = eigen.predict(val_x[i])
    if cls == test_y[i]:
        test_hit += 1
    else:
        test_miss += 1

print 'lbph+eigen validation error'
print 'error rate:', str(round((float(miss)/float(hit+miss))*100.0, 2))+'%'

print 'lbph+eigen testing error'
print 'error rate:', str(round((float(test_miss)/float(test_hit+test_miss))*100.0, 2))+'%'
"""

