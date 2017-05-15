import dlib
import os
import glob
import numpy as np
import cv2

mean_file = None
db_path = 'dataset'
db = None

#  file names
if not os.path.exists(db_path):
    print('Database path is not existed!')
folders = sorted(glob.glob(os.path.join(db_path, '*')))

for name in folders:
    print('loading {}:'.format(name))
    img_list = glob.glob(os.path.join(name, '*.jpg'))
    for img_name in img_list:
        print img_name
        img = cv2.imread(img_name)
        #   Crop faces
        face_detector = dlib.get_frontal_face_detector()
        ldmark_detector = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        det = face_detector(img, 0)
        for k, d in enumerate(det):
            landmarks1 = []
            shape1 = ldmark_detector(img, d)
            landmarks1 = [(shape1.part(i).x, shape1.part(i).y) for i in range(68)]
            eye_l = np.mean(landmarks1[36:42], axis=0)
            eye_r = np.mean(landmarks1[42:48], axis=0)
        crop_face = np.copy(img[max(0, d.top()):d.bottom(), max(0, d.left()):d.right(), :])
        cv2.imwrite(img_name, crop_face)