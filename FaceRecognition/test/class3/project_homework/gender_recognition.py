from PyQt4 import QtCore
from caffe_net import *
import cv2


class Gender_recognizer(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Gender_recognizer, self).__init__()
        caffemodel = './deep_model/ez_gender.caffemodel'
        deploy_file = './deep_model/ez_gender.prototxt'
        mean_file = None
        self.net = Deep_net(caffemodel, deploy_file, mean_file, gpu=True)
        self.recognizing = False
        self.textBrowser = textBrowser
        self.label = ['Female', 'Male']

    def gender_recognition(self, face_info):
        if self.recognizing:
            img = []
            cord = []

            #TODO collect images from face_info


            if len(img) != 0:
                #TODO call deep learning for classfication
                
                #TODO writ on GUI
                
                #TODO emit signal when detection finished
                

    def startstopgender(self, checkbox):
        if checkbox.isChecked():
            self.recognizing = True
        else:
            self.recognizing = False



