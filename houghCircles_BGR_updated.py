'''
Reference
https://076923.github.io/posts/Python-opencv-29/
https://vmpo.tistory.com/35
'''

import cv2
import numpy as np
import time


class HoughCircles_BGR_updated():
    def capCam(self, fileName):
        start_time = time.time()

        # load video from camera
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            # read frame
            success, frame = cap.read()

            if success:
#                print('Cam Detected?   ', success)
                # show frame
                cv2.imshow('Camera Window', frame)
                k = cv2.waitKey(33)

                if time.time() - start_time >= 2:
                    # if esc key pressed, capture and save image
                    cv2.IMREAD_UNCHANGED  # read image file including alpha channel
                    cv2.imwrite('.../AutoCellTester/captured_' + fileName + '.png', frame)
                    cap.release()

            else:
                break


    def pinDet(self, fileName):
        # Load captured image
        global src
        src = cv2.imread('.../AutoCellTester/captured_' + fileName + '.png')
        # src = cv2.getRotationMatrix2D((h, w), 90, 1)         # rotate image

        row_from = 0  # row = height
        row_to = 1040
        col_from = 178  # column = width
        col_to = 1740

        global crop
        crop = src[row_from: row_to, col_from: col_to]

        global h, w, channel
        h, w, channel = crop.shape  # size of image

        # image 전처리 (blur, grayscale)
        blr = cv2.medianBlur(crop, 7)
        blrgray = cv2.cvtColor(blr, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('blr gray', blrgray)

        global circles
        # HoughCircles(img, 검출 방법_2단계 허프변환, 해상도 비율, 최소 거리, 케니 엣지 thres 중 higher value, accumulator thres_높을수록 정확한 원 검출, minR, maxR
        circles = cv2.HoughCircles(blrgray, cv2.HOUGH_GRADIENT, 1, 20, param1=150, param2=8, minRadius=3,
                                   maxRadius=10)

        # Draw detected circles & indicate its rad (검출된 원의 개수: circles[0])
        if circles is None:
            print('Circles not detected!')
            exit()

        print(fileName + 'num of detected pins: ', circles.shape[1])  # 검출된 핀 개수 출력


    def pinStatusDet(self, fileName):
        # Load BGR value of pixels
        pix_bgr = np.array(crop)

        # 검출된 pin 하나씩 loop 순환
        for i in circles[0]:
            # 변수에 핀(원)의 좌표, 반지름 값 할당
            x = i[0]
            y = i[1]
            r = i[2]

            # 픽셀 값 추출할 위치 조정을 위한 상수 (클수록 실제 핀 중심점에 가까운 곳에서 추출)
            k = 1.7

            # down 핀을 더 잘 판별하기 위해 핀의 중심점보다 어두운 쪽에서 픽셀 추출
            # 제 1사분면에 위치한 핀 -> 왼쪽 아래 가장자리 픽셀 추출
            if (x >= w/2) and (y <= h/2):
                bgr_val = pix_bgr[int(y + r/k), int(x - r/k)]
            # 제 2사분면에 위치 -> 왼쪽 위 가장자리 픽셀
            elif (x >= w/2) and (y >= h/2):
                bgr_val = pix_bgr[int(y - r/k), int(x - r/k)]
            # 제 3사분면에 위치 -> 오른쪽 위 가장자리 픽셀
            elif (x <= w/2) and (y >= h/2):
                bgr_val = pix_bgr[int(y - r/k), int(x + r/k)]
            # 제 4사분면에 위치 -> 오른쪽 아래 가장자리 픽셀
            else:
                bgr_val = pix_bgr[int(y + r/k), int(x + r/k)]


            # 각 핀의 좌표 값과 hsv 값 출력
            print('coordinate: ', (i[0], i[1]), '   bgr: ', bgr_val)

            # up인 pin은 초록색, down인 pin은 빨간색으로 표현
            if bgr_val[0] >= 220 or bgr_val[1] >= 220 or bgr_val[2] >= 220:  # up으로 판별 (the brighter, the higher 명도value)
                cv2.circle(crop, (int(x), int(y)), radius=int(r), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            else:  # down으로 판별
                cv2.circle(crop, (int(x), int(y)), radius=int(r), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        '''
           cv2.putText(crop, str(hsv_val), (int(i[0]-70), i[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)    # 이미지에 hsv 값 출력
           cv2.putText(crop, str(i[2]), (int(i[0]-50), i[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2, cv2.LINE_AA)    # 원의 반지름 값 출력
        '''

        # save image
        cv2.imwrite('.../AutoCellTester/captured and detected_' + fileName + '.png', crop)

        '''
        # show image
        cv2.imshow('Pin Detection_by HSV', crop)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''
