import cv2
import houghCircles_resize
import pattern_control
import os

pc = pattern_control.Pattern_Control()
h = houghCircles_resize.HoughCircles_HSV()


# Get 4 coordinates to crop (왼쪽 위부터 반시계 방향으로 pt1~4)
pc.pattern('.../AutoCellTester/dtm/1.dtm')
h.capCam('1.dtm')
pt1, pt2, pt3, pt4 = h.getCooCrop()


# Detect pin's status for every pattern
folderPath = '.../AutoCellTester/dtm/'
file_list = os.listdir(folderPath)

for fileName in file_list:
    pc.pattern('.../AutoCellTester/dtm/' + fileName)
    h.capCam(fileName)
    h.pinDet(fileName, pt1, pt2, pt3, pt4)
    h.pinStatusDet(fileName)

print('MISSION COMPLETE')
cv2.destroyAllWindows()
