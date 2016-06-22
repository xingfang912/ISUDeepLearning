import cv2
import DataMungingUtil
import time
import numpy as np

#path = '/home/ryan/datasets/CroppedYale_resized_pad_224'
#path = '/home/ryan/datasets/iPhone5_v2_resized_pad_224'
#path = '/home/ryan/datasets/SamsungGalaxyS4_resized_pad_224'
#path = '/home/ryan/datasets/iPhone5_plus_SamsungGalaxyS4_pad_224'
path = '/home/ryan/datasets/iPhone5_SamsungGalaxyS4_SamsungGalaxyTab2_resize_pad_224'
#path = '/home/ryan/datasets/Cropped_Yale_Subset_Periocular_resized_pad_224'

num_classes, ((train_x, train_y), (val_x, val_y), (test_x, test_y)) = DataMungingUtil.load_dataset(path, colorspace='gray', normalize=False)

eigen = cv2.createEigenFaceRecognizer()
lbph = cv2.createLBPHFaceRecognizer()

start = time.time()
#eigen.train(train_x, train_y)
lbph.train(train_x, train_y)
print 'training took', time.time() - start, 'seconds'

miss_hist = {}

hit, miss = 0, 0
for i in range(0, len(val_x)):
    cls, conf = lbph.predict(val_x[i])
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
    cls, conf = lbph.predict(test_x[i])
    if cls == test_y[i]:
        test_hit += 1
    else:
        test_miss += 1

print 'lbph validation error'
print 'error rate:', str(round((float(miss)/float(hit+miss))*100.0, 2))+'%'

print 'lbph testing error'
print 'error rate:', str(round((float(test_miss)/float(test_hit+test_miss))*100.0, 2))+'%'

print 'miss_hist:'
print miss_hist

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


