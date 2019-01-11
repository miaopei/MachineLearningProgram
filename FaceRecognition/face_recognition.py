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
        if not os.path.exists(self.db_path):
            print('Database path is not existed!')
        folders = sorted(glob.glob(os.path.join(self.db_path, '*')))
        for name in folders:

            print('loading {}:'.format(name))
            self.label.append(os.path.basename(name))
            img_list = glob.glob(os.path.join(name, '*.jpg'))
            
            imgs = [cv2.imread(img) for img in img_list]
            scores, pred_labels, fea = self.net.classify(imgs, layer_name='fc7')

            #print('fea.shape {}'.format(fea.shape))
            fea = np.mean(fea, 0)
            print(fea[:])
            if self.db is None:
                self.db = fea.copy()
            else:
                self.db = np.vstack((self.db, fea.copy()))
            

            #print fea
            print('done')
        print self.label

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
                # call deep learning for classfication
                prob, pred, fea = self.net.classify(img, layer_name='fc7')

                #print(fea)

                # search from db find the closest
                dist = sklearn.metrics.pairwise.cosine_similarity(fea, self.db)
                #print('dist = {}'.format(dist))
                pred = np.argmax(dist, 1)
                dist = np.max(dist, 1)

                #print('pred = {}'.format(pred))
                #print('maxdist = {}'.format(dist))
                #print('threshold = {}'.format(self.threshold/100.0))
                pred = [0 if dist[i] < self.threshold/100.0 else pred[i]+1 for i in range(len(dist)) ]

                #print('pred(after threshold) = {}'.format(pred))

                # writ on GUI

                msg = QtCore.QString("Face Recognition Pred: <span style='color:red'>{}</span>".format(' '.join([self.label[x] for x in pred])))
                self.textBrowser.append(msg)
                # emit signal when detection finished
                self.emit(QtCore.SIGNAL('face_id(PyQt_PyObject)'), [pred, cord])

    def set_threshold(self, th):
        self.threshold = th
        self.textBrowser.append('Threshold is changed to: {}'.format(self.threshold))

    def startstopfacerecognize(self, checkbox):
        if checkbox.isChecked():
            self.recognizing = True
        else:
            self.recognizing = False



