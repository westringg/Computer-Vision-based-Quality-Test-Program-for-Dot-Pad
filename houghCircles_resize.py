'''
Image size 2.5배 resizing한 뒤 houghcircles 처리하는 version
'''


import cv2
import numpy as np
import time


class HoughCircles_HSV():
    # Capture Image
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
                    cv2.imwrite('.../AutoCellTester/byCir_resize/captured_%s.png'%fileName, frame)
                    cap.release()

            else:
                break


    def getCooCrop(self):
        # Get four coordinates to crop captured image
        for i in range(1, 5):
            # load index img and original img
            idx = cv2.imread('.../AutoCellTester/byCir_resize/index%d.png' % i)
            grayIdx = cv2.cvtColor(idx, cv2.COLOR_BGR2GRAY)
            ori = cv2.imread('.../AutoCellTester/byCir_resize/captured_1.dtm.png')
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
            cv2.imwrite('.../AutoCellTester/byCir_resize/BFMatcher_SIFT_Homography%d.jpg' % i, res)


        return pt1, pt2, pt3, pt4



    def pinDet(self, fileName, pt1, pt2, pt3, pt4):
        # Detect every location of pins

        # initialize .txt file
        with open('.../AutoCellTester/writtenDtm/writtenDtm%s.txt' % fileName, 'w') as f:
            f.write('')


        # load original image
        global src
        src = cv2.imread('.../AutoCellTester/byCir_resize/captured_%s.png'% fileName)


        # crop image and show cropped img
        global crop
        mask = np.zeros(src.shape[0:2], dtype=np.uint8)
        points = np.array([[pt1, pt2, pt3, pt4]]).astype(np.int32)

        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

        crop = cv2.bitwise_and(src, src, mask=mask)
        crop = cv2.resize(crop, dsize=(0,0), fx=2.5, fy=2.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite('.../AutoCellTester/byCir_resize/captured and cropped_%s.png' % fileName, crop)

        # size of the cropped image
        global h, w, channel
        h, w, channel = crop.shape
        print('shape:   ', crop.shape)

        # image 전처리 (blur, grayscale, global thresholding)
        blr = cv2.medianBlur(crop, 7)
        blrgray = cv2.cvtColor(blr, cv2.COLOR_BGR2GRAY)
        ret, glbThres = cv2.threshold(blrgray, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite('.../AutoCellTester/byCir_resize/glbThres_%s.png'%fileName, glbThres)


        # HoughCircles(img, 검출 방법_2단계 허프변환, 해상도 비율, 최소 거리, 케니 엣지 thres 중 higher value, accumulator thres_높을수록 정확한 원 검출, minR, maxR
        global circles
        circles = cv2.HoughCircles(glbThres, cv2.HOUGH_GRADIENT, 1, 50, param1=20, param2=1, minRadius=7,
                                   maxRadius=20)
        circles=circles[0]      # circles array의 최외곽 bracket을 벗김 (편의상)
        print('circles_shape_dim_size', circles.shape, circles.ndim, circles.size)

        # Delete circle that is not pin, if any
        global filtCircles
        filtCircles = circles

        i = 0
        while i < filtCircles.shape[0]:
            # 잘못 검출된 원이 셀에 근접하게 위치해 있음 -> pin detection에서 배제
            xCoor = filtCircles[i][0]
            yCoor = filtCircles[i][1]
            rad = filtCircles[i][2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            pix_hsv = np.array(hsv)

            # Set boarder to remove wrongly detected circles
            if pt1[1]*2.5 < pt4[1]*2.5:
                upper = pt1[1]*2.5
            else:
                upper = pt4[1]*2.5

            if pt2[1]*2.5 > pt3[1]*2.5:
                lower = pt2[1]*2.5
            else:
                lower = pt3[1]*2.5

            if pt1[0]*2.5 < pt2[0]*2.5:
                left = pt1[0]*2.5
            else:
                left = pt2[0]*2.5

            if pt4[0]*2.5 > pt3[0]*2.5:
                right = pt4[0]*2.5
            else:
                right = pt3[0]*2.5

            # Delete all pins out of boarder
            if yCoor - rad < upper or yCoor - rad < upper or yCoor + rad > lower or yCoor + rad > lower \
                    or xCoor - rad < left or xCoor - rad < left or xCoor + rad > right or xCoor + rad > right \
                    or hsv[int(yCoor), int(xCoor)][2] == 0:
#                cv2.circle(crop, (int(xCoor), int(yCoor)), radius=int(rad), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                print('delete   ', filtCircles[i], hsv[int(yCoor), int(xCoor)][2])
                filtCircles = np.delete(filtCircles, i, axis=0)
                i -= 1
            i += 1


        # if no circle is detected
        if filtCircles is None:
            print('Circles not detected!')
            exit()

        # sort circles according to their position (to write txt file)
        filtCircles = filtCircles[np.lexsort((filtCircles[:,0], filtCircles[:,1]))]
        #        filtCircles = np.lexsort([filtCircles[:,1], filtCircles[:,0]])
        #        filtCircles = filtCircles[np.transpose(filtCircles)[::-1]]
        #        filtCircles = filtCircles[np.lexsort(np.transpose(filtCircles)[::-1], axis=0)]

        print(fileName + 'num of detected pins: ', filtCircles.shape[0])  # 검출된 핀 개수 출력




    def pinStatusDet(self, fileName):
        # Determine status(up/down) for every pin detected

        # Load HSV value of pixels
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        pix_hsv = np.array(hsv)

        thres_hsv = 250        # 추출한 hsv 값에 대한 threhold 값
        thres_rad = 13          # 대상 핀이 up으로 감지되기 위한 minimum radius

        # 검출된 pin 하나씩 loop 순환
        for i in filtCircles:
            # 변수에 핀(원)의 좌표, 반지름 값 할당
            x = i[0]
            y = i[1]
            r = i[2]

            # 픽셀 값 추출할 위치 조정을 위한 상수 (클수록 실제 핀 중심점에 가까운 곳에서 추출)
            k = 2.7

            # down 핀을 더 잘 판별하기 위해 핀의 중심점보다 어두운 쪽에서 픽셀 추출
            # 제 1사분면에 위치한 핀 -> 왼쪽 아래 가장자리 픽셀 추출
            if (x >= w / 2) and (y < h / 2):
                hsv_val = pix_hsv[int(y + r / k), int(x - r / k)]
                cv2.line(crop, (int(x - r / k), int(y + r / k)), (int(x - r / k), int(y + r / k)), [255, 0, 0], 2,
                         cv2.LINE_AA)
            # 제 2사분면에 위치 -> 왼쪽 위 가장자리 픽셀
            elif (x >= w / 2) and (y >= h / 2):
                hsv_val = pix_hsv[int(y - r / k), int(x - r / k)]
                cv2.line(crop, (int(x - r / k), int(y - r / k)), (int(x - r / k), int(y - r / k)), [255, 0, 0], 2,
                         cv2.LINE_AA)
                # 제 3사분면에 위치 -> 오른쪽 위 가장자리 픽셀
            elif (x < w / 2) and (y >= h / 2):
                hsv_val = pix_hsv[int(y - r / k), int(x + r / k)]
                cv2.line(crop, (int(x + r / k), int(y - r / k)), (int(x + r / k), int(y - r / k)), [255, 0, 0], 2,
                         cv2.LINE_AA)
            # 제 4사분면에 위치 -> 오른쪽 아래 가장자리 픽셀
            else:
                hsv_val = pix_hsv[int(y + r / k), int(x + r / k)]
                cv2.line(crop, (int(x + r / k), int(y + r / k)), (int(x + r / k), int(y + r / k)), [255, 0, 0], 2,
                         cv2.LINE_AA)

            # up인 pin은 초록색, down인 pin은 빨간색으로 표현
            if hsv_val[2] >= thres_hsv and r >= thres_rad:  # 명도(Value)가 밝고 size가 큰 핀은 up으로 판별 (the brighter, the higher)
                cv2.circle(crop, (int(x), int(y)), radius=int(r), color=(0, 255, 0), thickness=1,
                           lineType=cv2.LINE_AA)
                # 각 핀의 좌표 값과 hsv 값, pin statuas 출력 (up이면 1 down이면 0)
                print('coordinate: ', (i[0], i[1]), 'radius  ', i[2], 'hsv: ', hsv_val, '1')

            else:  # down으로 판별
                cv2.circle(crop, (int(x), int(y)), radius=int(r), color=(0, 0, 255), thickness=1,
                           lineType=cv2.LINE_AA)
                # 각 핀의 좌표 값과 hsv 값, pin statuas 출력 (up이면 1 down이면 0)
                print('coordinate: ', (i[0], i[1]), 'radius  ', i[2], '   hsv: ', hsv_val, '0')


        # write txt file composed of 0(down) and 1(up)
        n=0
        for i in range(1, 41):
            ithCir = filtCircles[n:n+59]
            n+=59+1

            ithCir = ithCir[np.lexsort((ithCir[:,1], ithCir[:,0]))]
            print('%dthCir line detection start'%i, n)

            # 검출된 pin 하나씩 loop 순환
            for i in ithCir:
                # 변수에 핀(원)의 좌표, 반지름 값 할당
                x = i[0]
                y = i[1]
                r = i[2]

                # 픽셀 값 추출할 위치 조정을 위한 상수 (클수록 실제 핀 중심점에 가까운 곳에서 추출)
                k = 2.7

                # down 핀을 더 잘 판별하기 위해 핀의 중심점보다 어두운 쪽에서 픽셀 추출
                # 제 1사분면에 위치한 핀 -> 왼쪽 아래 가장자리 픽셀 추출
                if (x >= w / 2) and (y < h / 2):
                    hsv_val = pix_hsv[int(y + r / k), int(x - r / k)]
                    cv2.line(crop, (int(x - r / k), int(y + r / k)), (int(x - r / k), int(y + r / k)), [255, 0, 0], 2,
                             cv2.LINE_AA)
                # 제 2사분면에 위치 -> 왼쪽 위 가장자리 픽셀
                elif (x >= w / 2) and (y >= h / 2):
                    hsv_val = pix_hsv[int(y - r / k), int(x - r / k)]
                    cv2.line(crop, (int(x - r / k), int(y - r / k)), (int(x - r / k), int(y - r / k)), [255, 0, 0], 2,
                             cv2.LINE_AA)
                    # 제 3사분면에 위치 -> 오른쪽 위 가장자리 픽셀
                elif (x < w / 2) and (y >= h / 2):
                    hsv_val = pix_hsv[int(y - r / k), int(x + r / k)]
                    cv2.line(crop, (int(x + r / k), int(y - r / k)), (int(x + r / k), int(y - r / k)), [255, 0, 0], 2,
                             cv2.LINE_AA)
                # 제 4사분면에 위치 -> 오른쪽 아래 가장자리 픽셀
                else:
                    hsv_val = pix_hsv[int(y + r / k), int(x + r / k)]
                    cv2.line(crop, (int(x + r / k), int(y + r / k)), (int(x + r / k), int(y + r / k)), [255, 0, 0], 2,
                             cv2.LINE_AA)

                # up인 pin은 초록색, down인 pin은 빨간색으로 표현
                if hsv_val[2] >= thres_hsv and r >= thres_rad:  # 명도(Value)가 밝고 size가 큰 핀은 up으로 판별 (the brighter, the higher)
                    with open('.../AutoCellTester/writtenDtm/writtenDtm%s.txt'%fileName, 'a') as f:
                        f.write('1')

                else:  # down으로 판별
                    with open('.../AutoCellTester/writtenDtm/writtenDtm%s.txt'%fileName, 'a') as f:
                        f.write('0')

            # txt file 줄 바꿈 (한 줄씩_60pin씩 작성)
            with open('.../AutoCellTester/writtenDtm/writtenDtm%s.txt' % fileName, 'a') as f:
                f.write('\n')

        # save final image
        cv2.imwrite('.../AutoCellTester/byCir_resize/captured and detected_%s.png'%fileName, crop)
