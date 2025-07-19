
import numpy as np
import shutil
import time
import cv2
import sys
import os

import processing.binary_classifier_util as bc
import processing.frame_selector_util as fs
import processing.object_detector_util as od
import processing.keypoint_detector_util as kd



class RealtimePipeline:
    def realtime_test(self):

        gst_stream = "nvarguscamerasrc sensor_id=0 ! \
        'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1' ! \
        nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=540' ! \
        nvvidconv ! nvegltransform ! nveglglessink -e"
        capture = cv2.VideoCapture(gst_stream, cv2.CAP_GSTREAMER)

        if (capture.isOpened() == False):
            print("Error opening video stream or file")

        while capture.isOpened():
            ret, frame = capture.read()
            if ret == True:
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        capture.release()
        cv2.destroyAllWindows()









if __name__ == "__main__":
    realtime_pipeline = RealtimePipeline()
    realtime_pipeline.realtime_test()