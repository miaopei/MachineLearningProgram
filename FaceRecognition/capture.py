from PyQt4 import QtCore, QtGui
import cv2
import os


class Capture(QtCore.QThread):
    def __init__(self, deviceid, width=640, height=360):
        super(Capture, self).__init__()
        self.deviceid = deviceid
        self.frame = None
        self.timer_frm = QtCore.QTimer()
        self.timer = QtCore.QTimer()
        self.width = width
        self.height = height
        self.FPS = 25

        # Note: If self.devideid is string, it loads the video from filesys, if self.deviceid is int, it opens webcam
        self.cap = cv2.VideoCapture(self.deviceid)
        #self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        #self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 320)

        self.PROCESS_INTERVAL = 0.001 # freq of processing 1 frame (seconds)
        self.capturing = False
        self.timer_frm.stop()
        self.timer.stop()

    def run(self):
        """
        Thread run method, set timer for both capture and processing
        :return:
        """
        self.timer_frm.timeout.connect(self.get_cv_frame)
        self.timer_frm.stop()
        #self.timer_frm.start(1000 / self.FPS)
        self.timer.timeout.connect(self.send_frame)
        self.timer.stop()
        #self.timer.start(1000 * self.PROCESS_INTERVAL)

    def __del__(self):
        self.cap.release()

    def startCapture(self):
        if not self.timer_frm.isActive():
            self.timer_frm.start(1000 / self.FPS)
            self.timer.start(1000 * self.PROCESS_INTERVAL)
        self.capturing = True

    def endCapture(self):
        self.capturing = False
        self.timer_frm.stop()
        self.timer.stop()

    def quitCapture(self):
        self.cap.release()
        QtCore.QCoreApplication.quit()

    def send_frame(self):
        """
        send frame to backend for processing
        :return: None
        """
        self.emit(QtCore.SIGNAL('getFrame(PyQt_PyObject)'), self.frame)

    def get_cv_frame(self):
        """
        Get one frame from cv video capture, convert to QImage and send Signal
        :return: None
        """
        if self.capturing:
             _, self.frame = self.cap.read()
        #     image = QtGui.QImage(self.frame.tostring(), self.frame.shape[1], self.frame.shape[0],
        #                      QtGui.QImage.Format_RGB888).rgbSwapped()
        #
        #     self.emit(QtCore.SIGNAL('newImage(QImage)'), image)

