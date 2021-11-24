import cv2
import numpy as np
import time


class HoughCircles_HSV():
    def capCam(self, fileName):
        start_time = time.time()

        # load video from camera
        cap = cv2.VideoCapture(1)

        while cap.isOpened():
            # read frame
            success, frame = cap.read()

            if success:
#                print('Cam Detected?   ', success)
                # show frame
                cv2.imshow('Camera Window', frame)
                k = cv2.waitKey(33)

                if time.time() - start_time >= 2:
                    cv2.IMREAD_UNCHANGED  # read image file including alpha channel
                    cv2.imwrite('.../AutoCellTester/bySize/captured_%s.png' % fileName, frame)
                    cap.release()

            else:
                break


    def getCooCrop(self):
        for i in range(1, 5):
            # load index img and original img
            idx = cv2.imread('.../AutoCellTester/bySize/index%d.png' % i)
            grayIdx = cv2.cvtColor(idx, cv2.COLOR_BGR2GRAY)
            ori = cv2.imread('.../AutoCellTester/bySize/captured_1.dtm.png')
            grayOri = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

            # create SIFT descriptor detector
            detector = cv2.SIFT_create()
            # get keypoint and descriptor for each img
            kp1, desc1 = detector.detectAndCompute(grayIdx, None)
            kp2, desc2 = detector.detectAndCompute(grayOri, None)

            # create BFMatcher, choose algorithm and set cross check
            matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            # calculate matching
            matches = matcher.match(desc1, desc2)

            # sort result by ascending order of distance
            matches = sorted(matches, key=lambda x: x.distance)
            # get min and max distance
            min_dist, max_dist = matches[0].distance, matches[-1].distance
            # set threshold as 20% of min distance
            ratio = 0.2
            good_thresh = (max_dist - min_dist) * ratio + min_dist
            # classify only points with shorter distance than threshold
            good_matches = [m for m in matches if m.distance < good_thresh]
            print('matches:%d/%d, min:%.2f, max:%.2f, thres:%.2f' % (
                len(good_matches), len(matches), min_dist, max_dist, good_thresh))

            # get coordinates of target img using queryIdx of good matching points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            # get coordinates of original img using trainIdx of good matching points
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            # get Perspective Transformation(원근변환행렬) using RANSAC
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # create coordinates of area in size of index img
            h, w, = idx.shape[:2]
            pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])
            # Perspective Transformation of coordinates of index img
            dst = cv2.perspectiveTransform(pts, mtrx)

            # allocate coordinate of each point
            if i==1:
                pt1 = dst[2][0]
            elif i==2:
                pt2 = dst[3][0]
            elif i==3:
                pt3 = dst[0][0]
            else:
                pt4 = dst[1][0]

            # draw transformed coordinates on original img
            ori = cv2.polylines(ori, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # draw selected matching points
            matchesMask = mask.ravel().tolist()
            res = cv2.drawMatches(idx, kp1, ori, kp2, good_matches, None, matchesMask=matchesMask,
                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            res_size = cv2.resize(res, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

            # print accuracy
            accuracy = float(mask.sum()) / mask.size
            print("accuracy: %d/%d(%.2f%%)" % (mask.sum(), mask.size, accuracy))

            # show result
            cv2.imwrite('.../AutoCellTester/bySize/BFMatcher_SIFT_Homography%d.jpg' % i, res)


        return pt1, pt2, pt3, pt4



    def pinDet(self, fileName, pt1, pt2, pt3, pt4):
        global src
        src = cv2.imread('.../AutoCellTester/bySize/captured_%s.png' % fileName)

        global h, w, channel
        h, w, channel = src.shape  # size of image
        print('shape:   ', src.shape)

        # crop image and show cropped img
        global crop
        mask = np.zeros(src.shape[0:2], dtype=np.uint8)
        points = np.array([[pt1, pt2, pt3, pt4]]).astype(np.int32)

        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

        crop = cv2.bitwise_and(src, src, mask=mask)
        crop = cv2.resize(crop, dsize=(0,0), fx=2.5, fy=2.5, interpolation=cv2.INTER_AREA)

        # image 전처리 (blur, grayscale)
        blr = cv2.medianBlur(crop, 7)
        blrgray = cv2.cvtColor(blr, cv2.COLOR_BGR2GRAY)
        ret, glbThres = cv2.threshold(blrgray, 100, 240, cv2.THRESH_BINARY)
        cv2.imwrite('.../AutoCellTester/bySize/glbThres_%s.png' % fileName, glbThres)


        global circles
        # HoughCircles(img, 검출 방법_2단계 허프변환, 해상도 비율, 최소 거리, 케니 엣지 thres 중 higher value, accumulator thres_높을수록 정확한 원 검출, minR, maxR
        circles = cv2.HoughCircles(glbThres, cv2.HOUGH_GRADIENT, 1, 45, param1=50, param2=7, minRadius=5,
                                   maxRadius=20)
        print('circles_shape_dim_size', circles.shape, circles.ndim, circles.size, 'circles[0]  ', circles[0].shape)

        # Delete circle that is not pin, if any
        global filtCircles
        cirUpdate = False
        for i in range(0, circles.shape[1]):
            # 잘못 검출된 원이 셀에 근접하게 위치해 있음 -> pin detection에서 배제
            yCoor = circles[0][i][1]
            rad = circles[0][i][2]
            if yCoor-rad < pt1[1] or yCoor-rad < pt4[1] or yCoor+rad > pt2[1] or yCoor+rad > pt3[1]:
                print('delete   ', circles[0][i])
                fCircles = np.delete(circles[0], i, axis=0)
                cirUpdate = True
        if cirUpdate:
            print('cirUpdate')
            filtCircles = fCircles.reshape((1,) + fCircles.shape)
        else:
            filtCircles = circles

        # Draw detected circles & indicate its rad (검출된 원의 개수: circles[0])
        if filtCircles is None:
            print('Circles not detected!')
            exit()

        print(fileName + 'num of detected pins: ', filtCircles.shape[1])  # 검출된 핀 개수 출력


    def pinStatusDet(self, fileName):
        # 검출된 pin 하나씩 loop 순환
        for i in filtCircles[0]:
            # 변수에 핀(원)의 좌표, 반지름 값 할당
            x = i[0]
            y = i[1]
            r = i[2]

            # up인 pin은 초록색, down인 pin은 빨간색으로 표현
            if r >= 17:  # size가 큰 pin은 up으로 판별
                cv2.circle(crop, (int(x), int(y)), radius=int(r), color=(0, 255, 0), thickness=1,
                           lineType=cv2.LINE_AA)
                print('Up  ', i)
            else:  # 작은 pin은 down으로 판별
                cv2.circle(crop, (int(x), int(y)), radius=int(r), color=(0, 0, 255), thickness=1,
                           lineType=cv2.LINE_AA)
                print('Down  ', i)

        # save final image
        cv2.imwrite('.../AutoCellTester/bySize/captured and detected_%s.png' % fileName, crop)
