import sys

sys.path.append("/home/paperspace/Software/caffe-master/python")
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import os
import time
from google.protobuf import text_format
import matplotlib.pyplot as plt


class DeepLearningModel:
    def __init__(self, caffeModel, deployFile, meanFile=None, gpu=False, deviceId=0):
        """
        Intialize the class

        :param caffemodel: path to a .caffemodel file
        :param deploy_file: -- path to a .prorotxt file
        :param gpu: -- if true, use the GPU for inference
        :param device_id: -- gpu id default 0
        """
        os.environ['GLOG_minloglevel'] = '2'
        if gpu:
            caffe.set_device(deviceId)
            caffe.set_mode_gpu()
            print("GPU mode")
        else:
            caffe.set_mode_cpu()
            print("CPU mode")

        self.net = caffe.Net(deployFile, caffeModel, caffe.TEST)
        self.transformer = self.getTransformer(deployFile, meanFile)

    def getTransformer(self, deployFile, meanFile=None):
        """
        Returns an instance of caffe.io.Transformer
        :param deploy_file: path to a .prototxt file
        :param mean_file:   path to a .binaryproto file (default=None)
        :return: caffe.io.Transformer
        """
        network = caffe_pb2.NetParameter()
        with open(deployFile) as infile:
            text_format.Merge(infile.read(), network)
        if network.input_shape:
            dims = network.input_shape[0].dim
        else:
            dims = network.input_dim[:4]
        t = caffe.io.Transformer(inputs={'data': dims})
        t.set_transpose('data', (2, 0, 1))  # (channel, height, width)

        if dims[1] == 3:
            t.set_channel_swap('data', (2, 1, 0))

        if meanFile:
            with open(meanFile, 'rb') as infile:
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(infile.read())
                if blob.HasField('shape'):
                    blob_dims = blob.shape.dim
                    assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is %s' % blob.shape
                elif blob.HasField('num') and blob.HasField('channels') and blob.HasField('height') and blob.HasField(
                        'width'):
                    blob_dims = (blob.num, blob.channels, blob.height, blob.width)
                else:
                    raise ValueError('blob does not provide shape or 4d dimensions')

                # For mean file
                pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
                t.set_mean('data', pixel)
        else:
            # pixel = [104, 117, 123]
            pixel = [129, 104, 93]
            t.set_mean('data', np.array(pixel))

        return t

    def forwardPass(self, images, transformer, batchSize=1, layer=None):
        caffeImages = []
        for image in images:
            if image.ndim == 2:
                caffeImages.append(image[:, :, np.newaxis])
            else:
                caffeImages.append(image)

        caffeImages = np.array(caffeImages)
        dims = transformer.inputs['data'][1:]

        scores = None
        feature = None

        for chunk in [caffeImages[x:x + batchSize] for x in xrange(0, len(caffeImages), batchSize)]:
            new_shape = (len(chunk),) + tuple(dims)
            if self.net.blobs['data'].data.shape != new_shape:
                self.net.blobs['data'].reshape(*new_shape)
            for idx, img in enumerate(chunk):
                imageData = transformer.preprocess('data', img)
                self.net.blobs['data'].data[idx] = imageData
            output = self.net.forward()[self.net.outputs[-1]]

            if layer is not None:
                if feature is None:
                    feature = np.copy(self.net.blobs[layer].data)
                else:
                    feature = np.vstack((feature, self.net.blobs[layer].data))

            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))
            return scores, feature

    def classify(self, imageList, layerName=None):
        # load image list
        _, channels, height, width = self.transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)

        # classify_start_time = time.time()
        feature = None
        # scores = self.forward_pass([caffe.io.load_image(x) for x in image_list], self.transformer)
        scores, feature = self.forwardPass(imageList, self.transformer, batchSize=1, layer=layerName)
        # print 'Classification took %s seconds.' % (time.time() - classify_start_time)
        # print scores
        return scores, np.argmax(scores, 1), feature
