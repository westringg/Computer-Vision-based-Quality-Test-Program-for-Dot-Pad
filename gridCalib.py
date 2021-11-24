import cv2
import numpy as np
import pinDet_byGrid

h = pinDet_byGrid.HoughCircles_HSV()


src = cv2.imread('.../AutoCellTester/byGrid/captured_Snowman.dtm.png')

pt1, pt2, pt3, pt4 = h.getCooCrop()
print('crop coor ', pt1, pt2, pt3, pt4)

mask = np.zeros(src.shape[0:2], dtype=np.uint8)
points = np.array([[pt1, pt2, pt3, pt4]]).astype(np.int32)

cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

crop = cv2.bitwise_and(src, src, mask=mask)
cv2.imwrite('.../AutoCellTester/cropped_Snowman.dtm.png', crop)

#exit()

w = pt4[0] - pt1[0]
h = pt2[1] - pt1[1]
rows, cols, ch = crop.shape

pts1 = np.float32([pt1, pt2, pt4, pt3])
#pts2 = np.float32([pt1, [pt1[0], pt2[1]], [pt4[0], pt1[1]], [pt4[0], pt2[1]]])
pts2 = np.float32([pt1, [pt1[0], pt1[1] + h], [pt1[0]+w, pt1[1]], [pt1[0]+w, pt1[1]+h]])

M = cv2.getPerspectiveTransform(pts1,pts2)

pers = cv2.warpPerspective(crop,M,(cols, rows))


cv2.imwrite(".../AutoCellTester/persTrans_Snowman.dtm.png", pers)
print('transformed img saved!')


global pix_hsv
hsv = cv2.cvtColor(pers, cv2.COLOR_RGB2HSV)
pix_hsv = np.array(hsv)

for pt in (pt1, [pt1[0], pt1[1] + h], [pt1[0]+w, pt1[1]], [pt1[0]+w, pt1[1]+h]):
    cv2.line(pers, (int(pt[0]), int(pt[1])), (int(pt[0]), int(pt[1])), [0, 0, 255], 5, cv2.LINE_AA)


cell1pin1 = [int(pt1[0]) + 6, int(pt1[1]) + 5]
cell30pin1 = [int(pt1[0]+w) - 26, int(pt1[1]) + 5]
cell15pin1 = [cell30pin1[0] - 615, cell30pin1[1]]
cv2.line(pers, cell15pin1, cell15pin1, [0, 0, 255], 2, cv2.LINE_AA)

for i in range(10):
    lpin1x = cell1pin1[0]
    for n in range(15):
        lpin1 = [lpin1x, cell1pin1[1]]
        lpin1x += 44
        lpin1y = lpin1[1]
        for i in range(4):
            lpin = [lpin1[0], lpin1y]
            print('hsv', lpin, pix_hsv[lpin[1], lpin[0]])
            cv2.line(pers, lpin, lpin, [0, 255, 0], 2, cv2.LINE_AA)
            lpin1y += 22
        lpin1y = lpin1[1]
        for i in range(4):
            lpin = [lpin1[0] + 20, lpin1y]
            print('hsv', lpin, pix_hsv[lpin[1], lpin[0]])
            cv2.line(pers, lpin, lpin, [0, 255, 0], 2, cv2.LINE_AA)
            lpin1y += 22

    rpin1x = cell15pin1[0]
    for n in range(15):
        rpin1 = [rpin1x, cell15pin1[1]]
        rpin1x += 44
        rpin1y = rpin1[1]
        for i in range(4):
            rpin = [rpin1[0], rpin1y]
            print('hsv', rpin, pix_hsv[rpin[1], rpin[0]])
            cv2.line(pers, rpin, rpin, [0, 255, 0], 2, cv2.LINE_AA)
            rpin1y += 22
        rpin1y = rpin1[1]
        for i in range(4):
            rpin = [rpin1[0] + 20, rpin1y]
            print('hsv', rpin, pix_hsv[rpin[1], rpin[0]])
            cv2.line(pers, rpin, rpin, [0, 255, 0], 2, cv2.LINE_AA)
            rpin1y += 22

    cell1pin1[1] += 89
    cell15pin1[1] += 89


cv2.imwrite('.../AutoCellTester/croppedCalib_Snowman.dtm.png', pers)
print('calibration done!')
