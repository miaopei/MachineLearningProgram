import dlib
from PyQt4 import QtCore
import MyGui
import numpy as np
import time
import cv2


class Face_detector(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_detector, self).__init__()
        self.face_detector = dlib.get_frontal_face_detector()
        self.ldmark_detector = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
        self.face_info = {}
        self.textBrowser = textBrowser
        self.detecting = True  # flag of if detect face
        self.ldmarking = False  # flag of if detect landmark
        self.total = 0

    def detect_face(self, img):
        if self.detecting:
            self.face_info = {}

            #det_start_time = time.time()

            #TODO call dlib detector
            

            #print 'Detection took %s seconds.' % (time.time() - det_start_time)


            #print('Number of face detected: {}'.format(len(dets)))
            if len(dets) > 0:
                self.textBrowser.append('Number of face detected: {}'.format(len(dets)))



            #TODO for each face do landmark detection and save face locations and landmarks and cropped faces


                self.face_info[k] = ([d.left(), d.top(), d.right(), d.bottom()], landmarks[18:], crop_face)    # 0:18 are face counture


                
            #TODO emit signal when detection finished
            



    def startstopdet(self, checkbox):
        if checkbox.isChecked():
            self.detecting = True
        else:
            self.detecting = False

    def startstopldmark(self, checkbox):
        if checkbox.isChecked():
            self.ldmarking = True
        else:
            self.ldmarking = False








