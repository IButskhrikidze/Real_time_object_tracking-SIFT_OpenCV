import cv2 as cv
import numpy as np
import argparse

cv_version = cv.__version__

if cv_version[0] == '4':
    print('please install opencv3 and then run program')
    exit(0)

arg = argparse.ArgumentParser()

arg.add_argument('-i', '--image', required=True, help='tracking image path')
arg.add_argument('-v', '--video', required=False, help='video path')
arg.add_argument('-o', '--output', required=False, help='output video path')

args = vars(arg.parse_args())

img = cv.imread(args['image'], cv.IMREAD_GRAYSCALE)

path = 0

if args['video'] != None:
    path = args['video']

cap = cv.VideoCapture(path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

if args['output'] != None:
    out = cv.VideoWriter(args['output'], cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (frame_width, frame_height))

sift = cv.xfeatures2d.SIFT_create()
kp_img, desk_img = sift.detectAndCompute(img, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp_grayframe, desk_grayframe = sift.detectAndCompute(grayframe, None)
    
    matches = flann.knnMatch(desk_img, desk_grayframe, k=2)
    
    good = []
    
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

# hompgraphy

    if len(good) > 10:
        qr_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        tr_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
        mat, masc = cv.findHomography(qr_pts, tr_pts, cv.RANSAC, 5.0)
        matches_mask = masc.ravel().tolist()
        
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h, ], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, mat)
        
        homo = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        frame = homo
    else:
        frame = grayframe
    
    cv.imshow('Video_cap', frame)
    if args['output'] != None:
        out.write(frame)
    
    key = cv.waitKey(1)

    if key > 1:
        break

cap.release()
out.release()
cv.destroyAllWindows()
