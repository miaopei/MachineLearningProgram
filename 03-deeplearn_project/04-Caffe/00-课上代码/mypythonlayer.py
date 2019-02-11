import sys
caffe_root='/home/tyd/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np                                                                                                                                                       
import yaml
import cv2   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

class MyLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.num = yaml.load(self.param_str)["num"]
        print "Parameter num : ", self.num

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].reshape(*bottom[0].shape)
        print bottom[0].data.shape
        print bottom[0].data
        top[0].data[...] = bottom[0].data + self.num
        print top[0].data[...]

    def backward(self, top, propagate_down, bottom):
        pass



net = caffe.Net('conv.prototxt',caffe.TEST)
#im = np.array(Image.open('timg.jpeg'))
im = np.array(cv2.imread('timg.jpeg'))
print im.shape
#im_input = im[np.newaxis, np.newaxis, :, :]
im_input = im[np.newaxis, :, :]
print im_input.shape
#print im_input.transpose((1,0,2,3)).shape
im_input2 = im_input.transpose((0,3,1,2))
print im_input2.shape
#print im_input.shape
net.blobs['data'].reshape(*im_input2.shape)
net.blobs['data'].data[...] = im_input2
net.forward()


