import cv2
from time import time
from Recognition import isVideo

class Camera:
    def __init__(self, filename):
        self.projectPath = '/home/paperspace/PycharmProjects/video-streaming/'
        self.videoPath = filename
        self.capture = cv2.VideoCapture(self.projectPath + self.videoPath)
        self.numFrames = self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        print 'numFrames:' + str(self.numFrames)
        self.frames = [open(self.projectPath + f + '.jpg', 'rb').read() for f in ['4', '5', '6']]
        self.count = 0

    def getFrame(self):
        flag, frame = self.capture.read()
        self.count += 1
        print self.count
        if self.count == self.numFrames:
            self.count = 0
            self.capture = cv2.VideoCapture(self.projectPath + self.videoPath)
        if frame is None:
            pass
            return self.frames[int(time()) % 3]
        else:
            cv2.imshow('video', frame)
            cv2.waitKey(10)
            res, image = cv2.imencode('.jpeg', frame)
        return image.tostring()


    def __del__(self):
        self.capture.release()

if __name__ == '__main__':
    camera = Camera('uploads/flash.avi')
    while True:
        frame = camera.getFrame()


