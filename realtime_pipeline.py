#TODO: update readme again
#TODO: cite the code sections you have used formally (gst, gfg etc)
import numpy as np
import shutil
import time
import cv2
import sys
import os
import subprocess

#TODO: After functional testing attempt to 'preload' model/s
import processing.object_detector_util as od
import processing.keypoint_detector_util as kd



class RealtimePipeline:
    def __init__(self):
        #Forces os's primary display (negates issues arising via ssh given commands)
        os.environ['DISPLAY'] = ':0'
        #Initial test approach
        # self.gst_stream = ["gst-launch-1.0", "nvarguscamerasrc", "!", "nvvidconv", "!", "nvegltransform", "!",
        #               "nveglglessink", "-e"]

        # self.gst_stream = "nvarguscamerasrc sensor_id=0, !, video/x-raw(memory:NVMM),width=1280, height=720,framerate=30/1, !, nvvidconv flip-method=0, !, video/x-raw,width=640, height=360, !, nvvidconv, !, video/x-raw,format=BGS, !, appsink -e"
        # self.gst_stream = "nvarguscamerasrc ! nvvidconv ! video/x-raw,format=BGR ! appsink"
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e"


    def process(self):
        # make sure its empty - (FROM PIPELINE)
        savepoint = "./processing/extracted_frames/"
        shutil.rmtree(savepoint)
        os.mkdir(savepoint)
        #Here leads to later fail
        capture = cv2.VideoCapture(self.gst_stream, cv2.CAP_GSTREAMER)
        if capture.isOpened() == False:
            print("Unable to run gst_stream properly")
        # print("Camera Opened Successfully")

        try:
            while True:
                    ret, frame = capture.read()
                    cv2.imshow('frame', frame)
                    cv2.imwrite(f"{savepoint}0.png", frame)
                    try:
                        print("\nCropping to region of interest...")
                        roi_frames = od.process(savepoint)
                        print("ROI FRAMES: ", roi_frames.shape)
                        print("\nDetecting keypoints...")
                        coordinates = kd.process(roi_frames)
                        print("\n{}\n".format(coordinates))
                        print(coordinates.shape)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error has arisen due to: {e}")
        except Exception as e:
            print(f"Error has arisen due to: {e}")
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_pipeline = RealtimePipeline()
    realtime_pipeline.process()