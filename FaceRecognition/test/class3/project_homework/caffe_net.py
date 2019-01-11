import sys
sys.path.append("~/Desktop/caffe-master/python")
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import os
import time
from google.protobuf import text_format
import matplotlib.pyplot as plt


class Deep_net:

    def __init__(self, caffemodel, deploy_file, mean_file=None, gpu=False, device_id=0):
        """
        Intialize the class

        :param caffemodel: path to a .caffemodel file
        :param deploy_file: -- path to a .prorotxt file
        :param gpu: -- if true, use the GPU for inference
        :param device_id: -- gpu id default 0
        """
        os.environ['GLOG_minloglevel'] = '2'
        if gpu:
            caffe.set_device(device_id)
            caffe.set_mode_gpu()
            print("GPU mode")
        else:
            caffe.set_mode_cpu()
            print("CPU mode")

        self.net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
        self.transformer = self.get_transformer(deploy_file,mean_file)


    def get_transformer(self, deploy_file, mean_file=None):
        """
        Returns an instance of caffe.io.Transformer
        :param deploy_file: path to a .prototxt file
        :param mean_file:   path to a .binaryproto file (default=None)
        :return: caffe.io.Transformer
        """
        network = caffe_pb2.NetParameter()
        with open(deploy_file) as infile:
            text_format.Merge(infile.read(), network)
        if network.input_shape:
            dims = network.input_shape[0].dim
        else:
            dims = network.input_dim[:4]
        t = caffe.io.Transformer(inputs={'data':dims})
        t.set_transpose('data',(2,0,1))  # (channel, height, width)

        if dims[1] == 3:
            t.set_channel_swap('data',(2,1,0))

        if mean_file:
            with open(mean_file,'rb') as infile:
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(infile.read())
                if blob.HasField('shape'):
                    blob_dims = blob.shape.dim
                    assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is %s' % blob.shape
                elif blob.HasField('num') and blob.HasField('channels') and blob.HasField('height') and blob.HasField('width'):
                    blob_dims = (blob.num, blob.channels, blob.height, blob.width)
                else:
                    raise ValueError('blob does not provide shape or 4d dimensions')

                # For mean file
                pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
                t.set_mean('data', pixel)
        else:
            #pixel = [104, 117, 123]
            pixel = [129, 104, 93]
            t.set_mean('data', np.array(pixel))

        return t

    def forward_pass(self, images, transformer, batch_size=1, layer=None):
        caffe_images = []
        for image in images:
            if image.ndim == 2:
                caffe_images.append(image[:, :, np.newaxis])
            else:
                caffe_images.append(image)

        caffe_images = np.array(caffe_images)
        dims = transformer.inputs['data'][1:]

        scores = None
        fea = None

        for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
            new_shape = (len(chunk), ) + tuple(dims)
            if self.net.blobs['data'].data.shape != new_shape:
                self.net.blobs['data'].reshape(*new_shape)
            for idx, img in enumerate(chunk):
                image_data = transformer.preprocess('data', img)
                self.net.blobs['data'].data[idx] = image_data
            output = self.net.forward()[self.net.outputs[-1]]

            if layer is not None:
                if fea is None:
                    fea = np.copy(self.net.blobs[layer].data)
                else:
                    fea = np.vstack((fea, self.net.blobs[layer].data))

            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))
        return scores, fea


    def classify(self, image_list, layer_name=None):
        #load image list
        _, channels, height, width = self.transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)

        #classify_start_time = time.time()
        fea = None
        #scores = self.forward_pass([caffe.io.load_image(x) for x in image_list], self.transformer)
        scores, fea = self.forward_pass(image_list, self.transformer, batch_size=1, layer=layer_name)
        #print 'Classification took %s seconds.' % (time.time() - classify_start_time)
        #print scores
        return scores, np.argmax(scores,1), fea

    def test(self):
        #img_list = ['./test_img/face1.jpg','./test_img/face3.jpg']

        img_list = ['./db/Murphy/871.jpg', './db/Neo/374.jpg','./db/Red/642.jpg']

        import dlib
        import cv2
        face_detector = dlib.get_frontal_face_detector()
        imgs = []
        for f in img_list:
            img = cv2.imread(f)
            #dets = face_detector(img)
            #for d in dets:
            #    imgs.append(img[d.left():d.right(), d.top():d.bottom(),:])
            imgs.append(img)


        scores, pred_labels, fea = self.classify(imgs, layer_name='fc6')

        print fea

        print(scores)
        print(pred_labels)


    def showimage(self, im):
        if im.ndim == 3:
            im = im[:, :, ::-1]
        plt.set_cmap('jet')
        plt.imshow(im)
        plt.show()

    def vis_square(self, data, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        self.showimage(data)


if __name__ == "__main__":
  
    caffemodel = './deep_model/VGG_FACE.caffemodel'
    deploy_file = './deep_model/VGG_FACE_deploy.prototxt'
    mean_file = None

    gpu = True
    net = Deep_net(caffemodel, deploy_file, mean_file,gpu)
    net.test()








