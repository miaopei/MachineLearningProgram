import numpy as np
import matplotlib.pyplot as plt
#import Image
import sys
import os
from  math import pow
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
import random
caffe_root = '/home/matt/Documents/caffe/'

sys.path.insert(0, caffe_root + 'python')
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()



class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Rect(object):
    def __init__(self, p1, p2):
        '''Store the top, bottom, left and right values for points
               p1 and p2 are the (corners) in either order
        '''
        self.left   = min(p1.x, p2.x)
        self.right  = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top    = max(p1.y, p2.y)

    def __str__(self):
        return "Rect[%d, %d, %d, %d]" % ( self.left, self.top, self.right, self.bottom )

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def rect_overlaps(r1,r2):
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)

def rect_merge(r1,r2, mergeThresh):
    # centralPt1 = Point((r1.left + r1.right)/2,(r1.top + r1.bottom)/2)
    # centralPt2 = Point((r2.left + r2.right)/2,(r2.top + r2.bottom)/2)
    if rect_overlaps(r1,r2):
        # dist = calculateDistance((r1.left + r1.right)/2, (r1.top + r1.bottom)/2, (r2.left + r2.right)/2, (r2.top + r2.bottom)/2)
        SI= abs(min(r1.right, r2.right) - max(r1.left, r2.left)) * abs(max(r1.bottom, r2.bottom) - min(r1.top, r2.top))
        SA = abs(r1.right - r1.left)*abs(r1.bottom - r1.top)
        SB = abs(r2.right - r2.left)*abs(r2.bottom - r2.top)
        S=SA+SB-SI
        ratio = float(SI) / float(S)
        if ratio > mergeThresh :
            return 1
    return 0







def generateBoundingBox(featureMap, scale):
    boundingBox = []
    stride = 32
    cellSize = 227
    #227 x 227 cell, stride=32
    for (x,y), prob in np.ndenumerate(featureMap):
        if(prob >= 0.95):
            print prob
            boundingBox.append([float(stride * y)/ scale, float(x * stride)/scale, float(stride * y + cellSize - 1)/scale, float(stride * x + cellSize - 1)/scale, prob])
    #sort by prob, from max to min.
    #boxes = np.array(boundingBox)
    return boundingBox

def nms_average(boxes, groupThresh=2, overlapThresh=0.2):
    rects = []
    temp_boxes = []
    weightslist = []
    new_rects = []
    #print 'boxes: ', boxes
    for i in range(len(boxes)):
        if boxes[i][4] > 0.2:
            rects.append([boxes[i,0], boxes[i,1], boxes[i,2]-boxes[i,0], boxes[i,3]-boxes[i,1]])
    #print 'rects: ', rects
    # for i in range(len(rects)):
    #     rects.append(rects[i])

    rects, weights = cv2.groupRectangles(rects, groupThresh, overlapThresh)
    #######################test#########
    rectangles = []
    for i in range(len(rects)):
        # A______
        # |      |
        # -------B

    #                       A                                       B
        testRect = Rect( Point(rects[i,0], rects[i,1]), Point(rects[i,0]+rects[i,2], rects[i,1]+rects[i,3]))
        rectangles.append(testRect)
    clusters = []
    for rect in rectangles:
        matched = 0
        for cluster in clusters:
            if (rect_merge( rect, cluster , 0.2) ):
                matched=1
                cluster.left   =  (cluster.left + rect.left   )/2
                cluster.right  = ( cluster.right+  rect.right  )/2
                cluster.top    = ( cluster.top+    rect.top    )/2
                cluster.bottom = ( cluster.bottom+ rect.bottom )/2

        if ( not matched ):
            clusters.append( rect )
    # print "Clusters:"
    # for c in clusters:
    #     print c
    ###################################
    result_boxes = []
    for i in range(len(clusters)):
        # result_boxes.append([rects[i,0], rects[i,1], rects[i,0]+rects[i,2], rects[i,1]+rects[i,3], 1])
        result_boxes.append([clusters[i].left, clusters[i].bottom, clusters[i].right, clusters[i].top, 1])
    #print 'result_boxes: ', result_boxes
    return result_boxes


def face_detection(imgFile):
    net_full_conv = caffe.Net(os.path.join(caffe_root, 'faceDetect', 'deploy_full_conv.prototxt'),
                              os.path.join(caffe_root, 'faceDetect', 'alexnet_iter_50000_full_conv.caffemodel'),
                              caffe.TEST)
    randNum = random.randint(1,10000)
    
    
    
    scales = []
    factor = 0.793700526
    
    #img = Image.open(imgFile.strip())
    #img = img.convert('RGB')
    
    img = cv2.imread(imgFile)
    print img.shape
    
    largest = min(2, 4000/max(img.shape[0:2]))
    scale = largest
    minD = largest*min(img.shape[0:2])
    while minD >= 227:
        scales.append(scale)
        scale *= factor
        minD *= factor

    total_boxes = []

    for scale in scales:
        #resize image
        scale_img = cv2.resize(img,((int(img.shape[0] * scale), int(img.shape[1] * scale))))
        cv2.imwrite('/home/admin01/scale_img.jpg',scale_img)
        #scale_img.save("tmp{0}.jpg".format(randNum))
        # load input and configure preprocessing
        #im = caffe.io.load_image("tmp{0}.jpg".format(randNum))
        im = caffe.io.load_image('/home/admin01/scale_img.jpg')
        net_full_conv.blobs['data'].reshape(1,3,scale_img.shape[1],scale_img.shape[0])
        transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)

        # make classification map by forward and print prediction indices at each location
        out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
        print out['prob'][0,1].shape
        #print out['prob'][0].argmax(axis=0)
        boxes = generateBoundingBox(out['prob'][0,1], scale)
        #plt.subplot(1, 2, 1)
        #plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
        #plt.subplot(1, 2, 2)
        #plt.imshow(out['prob'][0,1])
        #plt.show()
        #print boxes
        if(boxes):
            total_boxes.extend(boxes)

            # boxes_nms = np.array(total_boxes)
            # true_boxes = nms(boxes_nms, overlapThresh=0.3)
            # #display the nmx bounding box in  image.
            # draw = ImageDraw.Draw(scale_img)
            # for box in true_boxes:
            #     draw.rectangle((box[0], box[1], box[2], box[3]) )
            # scale_img.show()

    #nms
    print total_boxes
    boxes_nms = np.array(total_boxes)
    true_boxes = nms_average(boxes_nms, 1, 0.2)
    if not true_boxes == []:
        (x1, y1, x2, y2) = true_boxes[0][:-1]
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0))
        win = cv2.namedWindow('test win', flags=0)  
  
        cv2.imshow('test win', img)  
          
        cv2.waitKey(0)  
#                 x1 = int(max(1, x1-(x2-x1)/6))
#                 y1 = int(max(1, y1-(y2-y1)/3))
#                 x2 = int(min(img.size[0], x2+(x2-x1)/6))
#                 cvimg = cv2.imread(imgFile)
#                 if cvimg == None:
#                     continue
#                 cvimg = cvimg[y1:y2, x1: x2]
#                 cvimg = cv2.resize(cvimg, (256,256))
#                 outputPath = os.path.join(imgPath+'-c', folder, str(count)+'.jpg')
#                 cv2.imwrite(outputPath, cvimg)
#                 count += 1

if __name__ == "__main__":
    imgFile = '/home/matt/Documents/caffe/faceDetect/tmp1470.jpg'

    face_detection(imgFile)
