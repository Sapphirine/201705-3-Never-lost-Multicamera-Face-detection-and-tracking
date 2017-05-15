import cv2
import os
import glob
import types
import dlib
import skvideo.io
import numpy as np
import sklearn.metrics.pairwise
import datetime
from DeepLearningModel import *
from DetectionTest import detect_face, drawBoxes
import multiprocessing


def isVideo(filename):
    if (filename.rsplit('.', 1)[1] == 'avi'):
        return True
    return False

caffeModelFace = '/home/paperspace/PycharmProjects/video-streaming/models/VGG_FACE.caffemodel'
deployFileFace = '/home/paperspace/PycharmProjects/video-streaming/models/VGG_FACE_deploy.prototxt'
#caffeModelGender = '/home/paperspace/PycharmProjects/video-streaming/models/ez_gender.caffemodel'
#deployFileGender = '/home/paperspace/PycharmProjects/video-streaming/models/ez_gender.prototxt'
caffeModelAge = '/home/paperspace/PycharmProjects/video-streaming/models/age_net.caffemodel'
deployFileAge = '/home/paperspace/PycharmProjects/video-streaming/models/deploy_age.prototxt'
caffeModelGender = '/home/paperspace/PycharmProjects/video-streaming/models/gender_net.caffemodel'
deployFileGender = '/home/paperspace/PycharmProjects/video-streaming/models/deploy_gender.prototxt'
emotionMeanFile = '/home/paperspace/PycharmProjects/video-streaming/models/mean.binaryproto'
caffeModelEmo = '/home/paperspace/PycharmProjects/video-streaming/models/EmotiW_VGG_S.caffemodel'
deployFileEmo = '/home/paperspace/PycharmProjects/video-streaming/models/emo.prototxt'
caffeModelP = '/home/paperspace/PycharmProjects/video-streaming/models/det1.caffemodel'
deployFileP = '/home/paperspace/PycharmProjects/video-streaming/models/det1.prototxt'
caffeModelR = '/home/paperspace/PycharmProjects/video-streaming/models/det2.caffemodel'
deployFileR = '/home/paperspace/PycharmProjects/video-streaming/models/det2.prototxt'
caffeModelO = '/home/paperspace/PycharmProjects/video-streaming/models/det3.caffemodel'
deployFileO = '/home/paperspace/PycharmProjects/video-streaming/models/det3.prototxt'
meanFile = None
genderMeanFile = '/home/paperspace/PycharmProjects/video-streaming/models/agemean.binaryproto'
ageMeanFile = '/home/paperspace/PycharmProjects/video-streaming/models/agemean.binaryproto'

class VideoProcess:
    def __init__(self, filename):
        self.projectPath = '/home/paperspace/PycharmProjects/video-streaming/'
        self.filename = filename
        self.datasetPath = self.projectPath + 'dataset/'
        self.dataset = None
        self.faceInfo = {}
        self.landmarking = True
        self.faceLabel = ['Stranger']
        self.emoLabel = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral',  'Sad', 'Surprise']
        self.targetLabel = 'Sheldon'
        #self.genderLabel = ['Female', 'Male']  for ez_gender.caffemodel
        self.genderLabel = ['Male', 'Female']
        self.ageLabel = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        self.faceDetector = dlib.get_frontal_face_detector()
        self.landmarkDetector = dlib.shape_predictor(self.projectPath + 'models/shape_predictor_68_face_landmarks.dat')
        self.faceNet = DeepLearningModel(caffeModelFace, deployFileFace, meanFile, gpu=True)
        self.genderNet = DeepLearningModel(caffeModelGender, deployFileGender, meanFile, gpu=True)
        self.emoNet = DeepLearningModel(caffeModelEmo, deployFileEmo, emotionMeanFile, gpu=True)
        self.ageNet = DeepLearningModel(caffeModelAge, deployFileAge, ageMeanFile, gpu=True)
        self.PNet = caffe.Net(deployFileP, caffeModelP, caffe.TEST)
        self.RNet = caffe.Net(deployFileR, caffeModelR, caffe.TEST)
        self.ONet = caffe.Net(deployFileO, caffeModelO, caffe.TEST)
        self.outputFile = None
        self.tracker = None
        self.bodyTracker = None
        self.bodyFlag = False
        self.tracking = False
        self.cap = None
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.minsize = 20
        self.modelType = 1
        self.initFace = None
        self.initBody = None

    def run(self):
        if isVideo(self.filename):
            print 'valid'
            self.cap = skvideo.io.VideoCapture(self.projectPath + self.filename, (640, 480))
            date = datetime.datetime.now()
            timestamp = date.strftime("%Y-%m-%d-%H-%M-%S")
            fourcc = cv2.cv.CV_FOURCC(*'mp4v')
            self.outputFile = 'uploads/output' + timestamp + '.avi'
            #output = skvideo.io.VideoWriter(self.projectPath + 'uploads/output' + timestamp + '.avi')
            if self.cap.isOpened():
                while True:
                    flag, frame = self.cap.read()
                    if flag == True:
                        self.frameProcess(frame)
                        #output.write(frame)
                        cv2.imshow('video', frame)
                        cv2.waitKey(1)
                    else:
                        break

            # output.release()
            self.cap.release()
            cv2.destroyAllWindows()

    def frameProcess(self, frame):
        if self.tracking is False:
            detectedFaces = self.faceDetection2(frame)
            captureTarget = self.faceRecognition2(frame, detectedFaces)
            self.emotionRecognition2(frame, detectedFaces)
            self.genderRecognition2(frame, detectedFaces)
            self.ageRecognition2(frame, detectedFaces)
            if captureTarget is not None:
                self.tracking = True
                points = [captureTarget]
                person_box = self.face2person(captureTarget)
                tmp = [person_box[0][0], person_box[0][1], person_box[1][0], person_box[1][1]]
                self.tracker = [dlib.correlation_tracker() for _ in xrange(len(points))]
                [self.tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(points)]
                self.initFace = np.copy(frame[max(0, captureTarget[1]):captureTarget[3],
                                        max(0, captureTarget[0]):captureTarget[2], :])

                self.bodyFlag = True
                body = [tmp]
                self.bodyTracker = [dlib.correlation_tracker() for _ in xrange(len(body))]
                [self.bodyTracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(body)]
                self.initBody = np.copy(frame[max(0, tmp[1]):tmp[3],
                                        max(0, tmp[0]):tmp[2], :])
        else:
            self.faceTracking(frame)
            if self.tracking is False:
                self.tracker = None
                self.initFace = None
                
        if self.bodyFlag is True:
            self.bodyTracking(frame)
        else:
            self.bodyTracker = None
            self.initBody = None


    def face2person(self, target):
        x1 = target[0]
        x2 = target[1]
        x3 = target[2]
        x4 = target[3]
        person_x = [(3*x2)/2-(x4)/2, 7*x4-6*x2]
        person_y = [(3*(x1)/2-(x3)/2), (3*x3)/2-(x1)/2]
        person_box = zip(person_y, person_x)
        return person_box

    def frameTest(self, frame):
        if self.tracking is False:
            detectedFaces = self.faceDetection(frame)
            points, captureTarget = self.faceRecognition(frame, detectedFaces)

            # self.genderRecognition(frame, detectedFaces)
            # self.emotionRecognition(frame, detectedFaces)
            #cv2.imshow('Processed', frame)
            #cv2.waitKey(1)
            if len(detectedFaces) != 0:
            # if captureTarget is not None:
                self.tracking = True
                print 'start tracking'
                # points = self.convertCoord()
                # self.tracker = [dlib.correlation_tracker() for _ in xrange(len(points))]
                points = [points]
                print type(points)
                self.tracker = [dlib.correlation_tracker() for _ in xrange(len(points))]
                [self.tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(points)]
                flag = self.faceTracking()
        else:
            flag = self.faceTracking()
            if flag is False:
                self.tracking = False
                self.tracker = None

    def faceTracking(self, frame):
        for i in xrange(len(self.tracker)):
            self.tracker[i].update(frame)
            rect = self.tracker[i].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            print "Object {} tracked at [{}, {}] \r".format(i, pt1, pt2),
            if True:
                loc = (int(rect.left()), int(rect.top() - 20))
            txt = "Object tracked at [{}, {}]".format(pt1, pt2)
            cv2.putText(frame, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            if i == 0:
                # Compare the current bounding box with the original detected image for the matching score.
                p1x = pt1[0]
                p1y = pt1[1]
                p2x = pt2[0]
                p2y = pt2[1]

                img_init = []
                cropFace = self.initFace
                faceNorm = cropFace.astype(float)
                img_init.append(faceNorm)

                img_track = []
                cropFace = np.copy(frame[max(0, p1y):p2y, max(0, p1x):p2x, :])
                faceNorm = cropFace.astype(float)
                img_track.append(faceNorm)

                if len(img_init) != 0 and len(img_track) != 0:
                    prob, pred, feature_init = self.faceNet.classify(img_init, layerName='fc7')
                    prob, pred, feature_track = self.faceNet.classify(img_track, layerName='fc7')

                    dist = sklearn.metrics.pairwise.cosine_similarity(feature_track, self.dataset)
                    pred = np.argmax(dist, 1)
                    dist = np.max(dist, 1)

                    threshold = 0.2
                    if dist > threshold:
                        pred = pred + 1
                    else:
                        pred = 0
                if self.faceLabel[pred] != self.targetLabel:
                    self.tracking = False

                dist = sklearn.metrics.pairwise.cosine_similarity(feature_track, feature_init)
                if dist < 0.4:
                    self.tracking = False

    def bodyTracking(self, frame):
        if self.bodyTracker is None:
            print 'wtf'
        for i in xrange(len(self.bodyTracker)):
            self.bodyTracker[i].update(frame)
            rect = self.bodyTracker[i].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            loc = (int(rect.left()), int(rect.top() - 20))
            txt = "Object tracked at [{}, {}]".format(pt1, pt2)
            cv2.putText(frame, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)

            p1x = pt1[0]
            p1y = pt1[1]
            p2x = pt2[0]
            p2y = pt2[1]

            body_init = []
            cropBody = self.initBody
            cropBody = np.resize(cropBody, (100, 100, 3)).reshape((1, 30000))
            bodyNorm1 = cropBody.astype(float)
            body_init.append(bodyNorm1)


            body_track = []
            cropBody = np.copy(frame[max(0, p1y):p2y, max(0, p1x):p2x, :])
            cropBody = np.resize(cropBody, (100, 100, 3)).reshape((1, 30000))
            bodyNorm2 = cropBody.astype(float)
            body_track.append(bodyNorm2)

            dist = sklearn.metrics.pairwise.cosine_similarity(bodyNorm1, bodyNorm2)
            if dist < 0.5:
                self.bodyFlag = False

            #   Type here


    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3],[0.299,0.587,0.114])

    def genderRecognition(self, frame, detectedFaces):
        if len(detectedFaces) > 0:
            for k, face in self.faceInfo.items():
                img = []
                faceNorm = face[2].astype(float)
                img.append(faceNorm)
                if len(img) != 0:
                    prob, pred, fea = self.genderNet.classify(img)
                    cv2.putText(frame, self.genderLabel[pred], (face[0][0], face[0][1] - 10), 0, 1, (0, 255, 0), 3)
        return frame

    def genderRecognition2(self, frame, boundingboxes):
        for i in range(len(boundingboxes)):
            img = []
            cropFace = np.copy(frame[max(0, int(boundingboxes[i][1])):int(boundingboxes[i][3]),
                               max(0, int(boundingboxes[i][0])):int(boundingboxes[i][2]), :])
            faceNorm = cropFace.astype(float)
            img.append(faceNorm)
            if len(img) != 0:
                prob, pred, fea = self.genderNet.classify(img)
                cv2.putText(frame, self.genderLabel[pred], (int(boundingboxes[i][0]) - 50,
                                                             int(boundingboxes[i][1]) - 10), 0, 1, (0, 255, 0), 3)

    def ageRecognition2(self, frame, boundingboxes):
        for i in range(len(boundingboxes)):
            img = []
            cropFace = np.copy(frame[max(0, int(boundingboxes[i][1])):int(boundingboxes[i][3]),
                               max(0, int(boundingboxes[i][0])):int(boundingboxes[i][2]), :])
            faceNorm = cropFace.astype(float)
            img.append(faceNorm)
            if len(img) != 0:
                prob, pred, fea = self.ageNet.classify(img)
                cv2.putText(frame, self.ageLabel[pred], (int(boundingboxes[i][2]) + 80,
                                                             int(boundingboxes[i][3]) + 30), 0, 1, (0, 255, 0), 3)

    def emotionRecognition(self, frame, detectedFaces):
        if len(detectedFaces) > 0:
            for k, face in self.faceInfo.items():
                img = []
                faceNorm = face[2].astype(float)
                img.append(faceNorm)
                if len(img) != 0:
                    prob, pred, fea = self.emoNet.classify(img)
                    cv2.putText(frame, self.emoLabel[pred], (face[0][0] + 80, face[0][1] - 10), 0, 1, (0, 255, 0), 3)
                    print 'emotion label:', pred
        return frame

    def emotionRecognition2(self, frame, boundingboxes):
        for i in range(len(boundingboxes)):
            img = []
            cropFace = np.copy(frame[max(0, int(boundingboxes[i][1])):int(boundingboxes[i][3]),
                               max(0, int(boundingboxes[i][0])):int(boundingboxes[i][2]), :])
            faceNorm = cropFace.astype(float)
            img.append(faceNorm)
            if len(img) != 0:
                prob, pred, fea = self.emoNet.classify(img)
                cv2.putText(frame, self.emoLabel[pred], (int(boundingboxes[i][0]) + 80,
                                                             int(boundingboxes[i][1]) - 10), 0, 1, (0, 255, 0), 3)

    def faceDetection(self, frame):
        detectedFaces = self.faceDetector(frame, 0)
        for k, d in enumerate(detectedFaces):
            landmarks = []
            if self.landmarking:
                shape = self.landmarkDetector(frame, d)
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                eyeLeft = np.mean(landmarks[36:42], axis=0)
                eyeRight = np.mean(landmarks[42:48], axis=0)
            cropFace = np.copy(frame[max(0, d.top()):d.bottom(), max(0, d.left()):d.right(), :])
            self.faceInfo[k] = ([d.left(), d.top(), d.right(), d.bottom()], landmarks[18:], cropFace)
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 5)
        return detectedFaces

    def faceDetection2(self, frame):
        img_matlab = frame.copy()
        tmp = img_matlab[:, :, 2].copy()
        img_matlab[:, :, 2] = img_matlab[:, :, 0]
        img_matlab[:, :, 0] = tmp
        boxes, points = detect_face(img_matlab, self.minsize, self.PNet,
                                            self.RNet, self.ONet, self.threshold, False, self.factor)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        for i in range(x1.shape[0]):
            cv2.rectangle(frame, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0), 5)
            #print 'point is:', (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i]))
        return boxes

    def faceRecognition2(self, frame, boundingboxes):
        targetPos = None
        for i in range(len(boundingboxes)):
            img = []
            cropFace = np.copy(frame[max(0, int(boundingboxes[i][1])):int(boundingboxes[i][3]),
                               max(0, int(boundingboxes[i][0])):int(boundingboxes[i][2]), :])
            faceNorm = cropFace.astype(float)
            img.append(faceNorm)
            if len(img) != 0:
                prob, pred, feature = self.faceNet.classify(img, layerName='fc7')
                dist = sklearn.metrics.pairwise.cosine_similarity(feature, self.dataset)
                pred = np.argmax(dist, 1)
                dist = np.max(dist, 1)

                threshold = 0.2
                if dist > threshold:
                    pred = pred + 1
                else:
                    pred = 0
                cv2.putText(frame, self.faceLabel[pred], (int(boundingboxes[i][0]) - 80, int(boundingboxes[i][3])),
                            0, 1, (0, 255, 0), 3)

                if self.faceLabel[pred] == self.targetLabel:
                    targetPos = [int(boundingboxes[i][0]), int(boundingboxes[i][1]),
                                 int(boundingboxes[i][2]), int(boundingboxes[i][3])]

        return targetPos

    def faceRecognition(self, frame, detectedFaces):
        res = None
        points_detected = None
        if len(detectedFaces) > 0:
            for k, face in self.faceInfo.items():
                img = []
                faceNorm = face[2].astype(float)
                img.append(faceNorm)
                if len(img) != 0:
                    prob, pred, feature = self.faceNet.classify(img, layerName='fc7')
                    dist = sklearn.metrics.pairwise.cosine_similarity(feature, self.dataset)
                    pred = np.argmax(dist, 1)
                    dist = np.max(dist, 1)

                    threshold = 0.8
                    if dist > threshold:
                        pred = pred + 1
                    else:
                        pred = 0
                    cv2.putText(frame, self.faceLabel[pred], (face[0][0], face[0][3] + 30), 0, 1, (0, 255, 0), 3)

                    points_detected = face[0]
                    if self.faceLabel[pred] == self.targetLabel:
                        res = face[0]
        return points_detected, res
# Modified by lingyu for debugging, original one : return res

    def prepareDataset(self):
        if not os.path.exists(self.datasetPath):
            print('Database path is not existed!')
        folders = sorted(glob.glob(os.path.join(self.datasetPath, '*')))
        for name in folders:
            self.faceLabel.append(os.path.basename(name))
            imgList = glob.glob(os.path.join(name, '*.jpg'))

            imgs = [cv2.imread(img) for img in imgList]
            scores, predLabels, feature = self.faceNet.classify(imgs, layerName='fc7')

            feature = np.mean(feature, 0)
            print(feature[:])
            if self.dataset is None:
                self.dataset = feature.copy()
            else:
                self.dataset = np.vstack((self.dataset, feature.copy()))

    def getOutput(self):
        print self.outputFile
        return self.outputFile

def worker_1(interval):
    print 'worker_1'
    video = VideoProcess('uploads/tbbt.avi')
    video.prepareDataset()
    video.run()

def worker_2(interval):
    print 'worker_2'
    video = VideoProcess('uploads/visor_1.avi')
    video.prepareDataset()
    video.run()

def worker_3(interval):
    print 'worker_3'
    video = VideoProcess('uploads/visor_2.avi')
    video.prepareDataset()
    video.run()

def worker_4(interval):
    print 'worker_4'
    video = VideoProcess('uploads/bubu2.avi')
    video.prepareDataset()
    video.run()

if __name__ == '__main__':
    video = VideoProcess('uploads/tbbt.avi')
    video.prepareDataset()
    video.run()
    #p1 = multiprocessing.Process(target= worker_1, args=(2,))
    #p2 = multiprocessing.Process(target= worker_2, args=(3,))
    #p3 = multiprocessing.Process(target= worker_3, args=(4,))
    #p4 = multiprocessing.Process(target= worker_4, args=(4,))
    #p1.start()
    #p2.start()
    #p3.start()
    #p4.start()

    print 'Numver of CPU is: ' + str(multiprocessing.cpu_count())
