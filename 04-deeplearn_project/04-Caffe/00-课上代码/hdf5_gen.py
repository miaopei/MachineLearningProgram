import h5py
import os
import cv2
import math
import numpy as np
import random
import re

root_path = "/home/tyd/caffe_case/HDF5/image"

with open("/home/tyd/caffe_case/HDF5/hdf5.txt", 'r') as f:
    lines = f.readlines()

num = len(lines)
random.shuffle(lines)


imgAccu = 0
imgs = np.zeros([num, 3, 224, 224])
labels = np.zeros([num, 10])
for i in range(num):
    line = lines[i]
    segments = re.split('\s+', line)[:-1]
    print segments[0]
    img = cv2.imread(os.path.join(root_path, segments[0]))
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2,0,1)
    imgs[i,:,:,:] = img.astype(np.float32)
    for j in range(10):
        labels[i,j] = float(segments[j+1])*224/256

batchSize = 1
batchNum = int(math.ceil(1.0*num/batchSize))

imgsMean = np.mean(imgs, axis=0)
#imgs = (imgs - imgsMean)/255.0
labelsMean = np.mean(labels, axis=0)
labels = (labels - labelsMean)/10

if os.path.exists('trainlist.txt'):
    os.remove('trainlist.txt')
if os.path.exists('testlist.txt'):
    os.remove('testlist.txt')
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
for i in range(batchNum):
    start = i*batchSize
    end = min((i+1)*batchSize, num)
    if i < batchNum-1:
        filename = '/home/tyd/caffe_case/HDF5/h5/train{0}.h5'.format(i)
    else:
        filename = '/home/tyd/caffe_case/HDF5/h5/test{0}.h5'.format(i-batchNum+1)
    print filename
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data = np.array((imgs[start:end]-imgsMean)/255.0).astype(np.float32), **comp_kwargs)
        f.create_dataset('label', data = np.array(labels[start:end]).astype(np.float32), **comp_kwargs)

    if i < batchNum-1:
        with open('/home/tyd/caffe_case/HDF5/h5/trainlist.txt', 'a') as f:
            f.write(os.path.join(os.getcwd(), 'train{0}.h5').format(i) + '\n')
    else:
        with open('/home/tyd/caffe_case/HDF5/h5/testlist.txt', 'a') as f:
            f.write(os.path.join(os.getcwd(), 'test{0}.h5').format(i-batchNum+1) + '\n')

imgsMean = np.mean(imgsMean, axis=(1,2))
with open('mean.txt', 'w') as f:
    f.write(str(imgsMean[0]) + '\n' + str(imgsMean[1]) + '\n' + str(imgsMean[2]))
