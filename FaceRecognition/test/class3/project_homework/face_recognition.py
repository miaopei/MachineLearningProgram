from PyQt4 import QtCore
from caffe_net import *
import glob
import caffe
import sklearn.metrics.pairwise
import cv2

class Face_recognizer(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_recognizer, self).__init__()

        # load face model
        caffemodel = './deep_model/VGG_FACE.caffemodel'
        deploy_file = './deep_model/VGG_FACE_deploy.prototxt'
        mean_file = None
        self.net = Deep_net(caffemodel, deploy_file, mean_file, gpu=True)

        self.recognizing = True
        self.textBrowser = textBrowser
        self.threshold = 0
        self.label = ['Stranger']
        self.db_path = './db'
        #self.db = []
        self.db = None
        # load db
        self.load_db()


    def load_db(self):
        # TODO load gallery images from filesystem and store the labels and deep features in self.db



    def face_recognition(self, face_info):
        if self.recognizing:
            img = []
            cord = []
            for k, face in face_info[0].items():
                face_norm = face[2].astype(float)
                face_norm = cv2.resize(face_norm, (128, 128))
                img.append(face_norm)
                cord.append(face[0][0:2])

            if len(img) != 0:

                # TODO call deep learning for classfication
                


                # TODO search from db find the closest
                

                pred = np.argmax(dist, 1)
                dist = np.max(dist, 1)

                # TODO matching, find the matching ID


                # write on GUI
                msg = QtCore.QString("Face Recognition Pred: <span style='color:red'>{}</span>".format(' '.join([self.label[x] for x in pred])))
                self.textBrowser.append(msg)

                # TODO emit signal when detection finished
                

    def set_threshold(self, th):
        self.threshold = th
        self.textBrowser.append('Threshold is changed to: {}'.format(self.threshold))

    def startstopfacerecognize(self, checkbox):
        if checkbox.isChecked():
            self.recognizing = True
        else:
            self.recognizing = False



