'''
Created on Mar 22, 2016

@author: admin01
'''
import cv2
if __name__ == '__main__':
    result_path = '/home/admin01/tangyudi/cnn2013/'
    #Abbas_Kiarostami_0001.jpg 75 165 87 177 106.750000 108.250000 143.750000 108.750000 131.250000 127.250000 106.250000 155.250000 142.750000 155.250000
    img = cv2.imread('/home/admin01/workspace/deep_landmark/cnn-face-data/lfw_5590/Abbas_Kiarostami_0001.jpg')
    #75 165 87 177
    cv2.rectangle(img, (75, 87), (165, 177), (0,0,255), 2)
    #cv2.circle(img, (int(75), int(87)), 1, (0,255,0), -1)
    #cv2.circle(img, (int(168), int(177)), 1, (0,255,0), -1)
    cv2.imwrite(result_path+'test.jpg', img)
