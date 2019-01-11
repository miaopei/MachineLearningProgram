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
            for k, face in face_info[0].items():
                face_norm = face[2].astype(float)
                img.append(face_norm)
                cord.append(face[0][0:2])

            if len(img) != 0:
                # call deep learning for classfication
                prob, pred, fea = self.net.classify(img)
                # writ on GUI
                self.textBrowser.append("Gender Recognition: <span style='color:green'>{}</span>".format([self.label[x] for x in pred]))
                # emit signal when detection finished
                self.emit(QtCore.SIGNAL('gender(PyQt_PyObject)'), [pred, cord])

    def startstopgender(self, checkbox):
        if checkbox.isChecked():
            self.recognizing = True
        else:
            self.recognizing = False



