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
        serName = "/dev/cu.usbserial-AD0KCSH7"
        ser = serial.Serial(serName, 115200, timeout=1)
        return ser


    def reqCMD_CellDisplay(self, argDestID, argMode, argStartOffset, argLength, argData, serPort):
        # argDestID(Board Line ID), argMode(SEQ NUM_Graphic은 0x00), argStartOffset(시작할 셀_n번째), argLength(MCU 한 개 당 제어하는 셀 개수), argData(배열_all up은 0xff)
        txData = bytearray(10 + argLength)

        txData[0] = 0xAA
        txData[1] = 0x55
        txData[2] = ((argLength & 0xFF00) >> 8)  # Length High Byte
        txData[3] = (0x06 + (argLength & 0x00FF))  # Length Low Byte
        txData[4] = argDestID  # Destination ID
        txData[5] = 0x02  # Command-High
        txData[6] = 0x00  # Command-Low
        txData[7] = argMode  # Sequence Number
        txData[8] = argStartOffset  # Data[0] : StartOffset

        for i in range(argLength):
            txData[9 + i] = argData[i]

        txData[9 + argLength] = self.makeCheckSumData(txData)

        serPort.write(txData)
#        print('txData:  ', bytearray(txData))

    #    rx = serPort.readline()
    #    print('rx:   ', rx)


    # Check Sum
    def makeCheckSumData(self, argData):
        result = 0xA5

        for j in range(4, len(argData) - 1):
            result ^= argData[j]

#        print('CheckSumCalc:  ', result)
        return result


    def pattern(self, dtmPath):
        # open Port
        port = self.openSerialPort()

        # read dtm file
        path = dtmPath
        f = open(path, 'rb')
        f.seek(30)
        cellData = f.read()

        # extract cell data and send it to Pad (20 times)
        i = 1
        while i <= 300:
            perMcu = cellData[i-1:i-1+15]
            print('i: ', i, '   ', perMcu, '   index (', i-1, ':', i-1+15-1, ')')
            argArray = perMcu
            '''
            argArray = bytearray(15)
            for j in range(0, 14):
                argArray[j] = cellData[j]
            '''
            print(int((i - 1) / 15) + 1, 'th MCU argArray sent:   ', argArray)
            self.reqCMD_CellDisplay(int((i-1)/15) + 1, 0, 0, 15, argArray, port)
            time.sleep(0.1)
            i += 15
        f.close()

