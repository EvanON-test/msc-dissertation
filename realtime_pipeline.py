#TODO: update readme again
#TODO: cite the code sections you have used formally (gst, gfg etc)


import numpy as np
import shutil
import time
import cv2
import sys
import os
import subprocess
from threading import Thread, Event
import queue
import argparse

#TODO: NEXT - Implement a approach where models stay loaded (as opposed to loaded every function call)
import processing.object_detector_util as od
import processing.keypoint_detector_util as kd


class FrameCaptureThread(Thread):
    """Thread responsible for capturing and displaying frames. It inherits Thread in an aid to run as part
    of a separate thread to the processing"""

    def __init__(self, gst_stream, frame_queue, stop_event, process_every_n_frames=60):
        """Initialises the frame capture thread"""
        super().__init__()
        self.gst_stream = gst_stream
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.process_every_n_frames = process_every_n_frames

    def stop(self):
        """Stops the frame capture thread by setting the stop event """
        self.stop_event.set()

    def run(self):
        """Main capture logic that reads the frames and adds them to a queue to pbe processed.
        Continues until stop event is set"""
        capture = None
        try:
            capture = cv2.VideoCapture(self.gst_stream, cv2.CAP_GSTREAMER)
            if capture.isOpened() == False:
                print("Unable to run gst_stream properly")
                return

            frame_counter = 0


            while not self.stop_event.is_set():
                ret, frame = capture.read()
                cv2.imshow('Live Feed', frame)
                frame_counter += 1
                if frame_counter % self.process_every_n_frames == 0:
                    try:
                        self.frame_queue.put(frame.copy())
                    except Exception as e:
                        print(f"CAPTURE FRAMES: Error has arisen due to: {e}")
        except Exception as e:
            print(f"CAPTRUE THREAD: Error has arisen due to: {e}")
        finally:
            capture.release()
            cv2.destroyAllWindows()


class FrameProcessingThread(Thread):
    """Thread responsible processing frames through the pipeline. It inherits Thread in an aid to run as part
    of a separate thread to the processing"""
    def __init__(self, frame_queue, stop_event):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    def stop(self):
        """Stops the frame processing thread by setting the stop event """
        self.stop_event.set()

    def run(self):
        """Main processing logic that processes the frames in the queue. Modified OD and KD logic from original Pipeline.
         Continues until stop event is set"""
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=1)

                    print("\nCropping to region of interest...")
                    try:
                        roi_frames = od.process_realtime(frame)
                        print("OD processed successfully!")
                    except Exception as e:
                        print("Potential Error, skipping frame..." + str(e))
                        self.frame_queue.task_done()
                        continue
                    if roi_frames is None:
                        print("roi_frames is none....skipping frame")
                        self.frame_queue.task_done()
                        continue

                    print("ROI FRAMES: ", roi_frames.shape)
                    print("\nDetecting keypoints...")
                    coordinates = kd.process(roi_frames)
                    print("\n{}\n".format(coordinates))
                    print(coordinates.shape)

                    self.frame_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"PROCESS FRAMES: Error has arisen due to: {e}")
                    try:
                        self.frame_queue.task_done()
                    except Exception as e:
                        print("Error arisen due to: " + str(e))
        except Exception as e:
            print(f"PROCESS THREAD: Error has arisen due to: {e}")



class RealtimePipeline:
    """Main class for running the realtime pipeline of frame capture, display and processing.
    This includes managing the created cpature and processing threads"""
    def __init__(self, process_every_n_frames=60):
        #Forces os's primary display (negates issues arising via ssh given commands)
        os.environ['DISPLAY'] = ':0'
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e"
        self.process_every_n_frames = process_every_n_frames
        #Threading components
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = Event()
        self.capture_thread = None
        self.process_thread = None


    def stop(self):
        """Stops the pipeline and cleans up resources"""
        self.stop_event.set()
        self.capture_thread.join()
        self.process_thread.join()
        cv2.destroyAllWindows()


    def process(self):
        """Creates and starts both threads"""
        try:
            #creates both threads
            self.capture_thread = FrameCaptureThread(self.gst_stream, self.frame_queue, self.stop_event, self.process_every_n_frames)
            self.process_thread = FrameProcessingThread(self.frame_queue, self.stop_event)

            #starts both threads
            self.capture_thread.start()
            self.process_thread.start()

            #waits for threads to complete
            self.capture_thread.join()
            self.process_thread.join()

        except KeyboardInterrupt:
            print("Interrupted by Keyboard")
            self.stop()
        except Exception as e:
            print("Error occurred due to: " + str(e))
        finally:
            self.stop()

    @staticmethod
    def run(process_every_n_frames=60):
        try:
            pipeline = RealtimePipeline(process_every_n_frames)
            pipeline.process()
        except Exception as e:
            print("PIPELINE ERROR: occurred due to: " + str(e))




if __name__ == "__main__":
    # realtime_pipeline = RealtimePipeline()
    # realtime_pipeline.process()

    # An updated approach. Argparse approach means the number of runs can added to the cli command
    parser = argparse.ArgumentParser(
        description='Run a CV pipeline with camera capture and processing')
    parser.add_argument("--frames_interval", type=int, default=60, help="Process every N frmaes (60 default)")
    args = parser.parse_args()
    RealtimePipeline.run(process_every_n_frames=args.frames_interval)