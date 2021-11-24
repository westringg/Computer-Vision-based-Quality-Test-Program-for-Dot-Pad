import binascii

import serial
import serial.tools.list_ports as sp
import numpy
import time
import os


class Pattern_Control:
    def openSerialPort(self):
        list = sp.comports()
        connected = []

        # PC와 연결된 COM Port 정보 출력
        for i in list:
            connected.append(i.device)
        print("Connected COM ports: " + str(connected))

        # open할 COM Port 선택한 뒤 Serial return
        serName = "/dev/..."
        ser = serial.Serial(serName, 115200, timeout=1)
        return ser

    
        '''
        Code hidden to avoid copyright issues (May contain confidential data)
        '''


    def pattern(self, dtmPath):
        # open Port
        port = self.openSerialPort()

        # read dtm file
        path = dtmPath
        f = open(path, 'rb')
        
        '''
        Code hidden to avoid copyright issues (May contain confidential data)
        '''
        
        f.close()

