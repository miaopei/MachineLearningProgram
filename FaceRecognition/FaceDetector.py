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
            dets = self.face_detector(img, 0)
            #print 'Detection took %s seconds.' % (time.time() - det_start_time)


            #print('Number of face detected: {}'.format(len(dets)))
            if len(dets) > 0:
                self.textBrowser.append('Number of face detected: {}'.format(len(dets)))

            for k, d in enumerate(dets):
                #print("Detection {}: left: {} Top: {} Right: {} Bottom: {}".format(
                #    k, d.left(), d.top(), d.right(), d.bottom()
                #))
                #self.textBrowser.append("Detection {}: left: {} Top: {} Right: {} Bottom: {}".format(
                #    k, d.left(), d.top(), d.right(), d.bottom()))

                # ldmark detection
                landmarks = []
                if self.ldmarking:
                    shape = self.ldmark_detector(img, d)
                    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                    eye_l = np.mean(landmarks[36:42], axis=0)
                    eye_r = np.mean(landmarks[42:48], axis=0)
                    #print eye_l, eye_r
                    #self.textBrowser.append('eye_l {}:{}'.format(eye_l[0], eye_r[1]))
                crop_face = np.copy(img[max(0, d.top()):d.bottom(), max(0,d.left()):d.right(), :])

                #print crop_face.shape
                #print ('top , bottom, left, right = {}, {}, {}, {}'.format(d.top(), d.bottom(), d.left(), d.right()))

                #save crop face
                #cv2.imwrite('./db/{}.jpg'.format(self.total), crop_face.copy())
                self.total+=1

                self.face_info[k] = ([d.left(), d.top(), d.right(), d.bottom()], landmarks[18:], crop_face)    # 0:18 are face counture
            # emit signal when detection finished
            self.emit(QtCore.SIGNAL('det(PyQt_PyObject)'), [self.face_info, img])



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








