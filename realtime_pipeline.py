#TODO: DONT FORGET TO cite the code sections you have used formally (gst, gfg etc)


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
    """Thread responsible for capturing and displaying frames. It inherits Thread in an aid to run as a separate
     thread alongside processing thread. This thread handles continuous frame capture and displaying following hte defined cadence
     (default 60)"""

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
        """Main capture logic that reads the frames and adds them to a queue to be processed.
        Continues until stop event is set"""
        capture = None
        try:
            #Initialises camera capture utilising Gstreamer approach
            capture = cv2.VideoCapture(self.gst_stream, cv2.CAP_GSTREAMER)
            #Verifies camera opened succesfully
            if capture.isOpened() == False:
                print("Unable to run gst_stream properly")
                return

            #intilises frame counter
            frame_counter = 0

            #The main capture loop, continues until stop event is set
            while not self.stop_event.is_set():
                #reads next frame from camera
                ret, frame = capture.read()
                #displys the frame
                cv2.imshow('Live Feed', frame)

                #checks for user input. If q pressed initiates stop event
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break

                #incremnts frame counter and checks whether it should be processed by dividing it by its user defined cadence
                frame_counter += 1
                if frame_counter % self.process_every_n_frames == 0:
                    try:
                        #creates a copy of the frame for the processing thread to utilise
                        self.frame_queue.put(frame.copy())
                    except Exception as e:
                        #elaborates on error and allows the continuation of loop
                        print(f"CAPTURE FRAMES: Error has arisen due to: {e}")

        except Exception as e:
            print(f"CAPTRUE THREAD: Error has arisen due to: {e}")
        finally:
            #Resouces minimisation after loops have completed
            capture.release()
            cv2.destroyAllWindows()


class FrameProcessingThread(Thread):
    """Thread responsible processing frames through the pipeline. It inherits Thread in an aid to run as part
    of a separate thread to the frame capture thread. This thread handles retrieving the frames from the queue and
    the processing of the OD and KD models over those frames """
    def __init__(self, frame_queue, stop_event):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    def stop(self):
        """Stops the frame processing thread by setting the stop event """
        self.stop_event.set()

    def run(self):
        """Main processing logic that processes the frames in the queue. Retrieves fram from queue and proccesses it
        outputting the results. Continues until stop event is set"""
        try:
            #continues until stop event set
            while not self.stop_event.is_set():
                try:
                    #waits for frame from capture thread
                    frame = self.frame_queue.get(timeout=1)

                    print("\nCropping to region of interest...")
                    try:
                        #processes frame through object detector which outputs region of interest and confidence level
                        roi_frames, confidence = od.process_realtime(frame)
                        print(f"Frame processed successfully, confidence: {confidence:.2f}")
                        #Skips frames which have a low confidence level
                        #TODO: need to clarify what level is appropriate. Reminder: Initial runs were showing 0.3 - 0.6ish
                        if confidence < 0.7:
                            print("Confidence too low: " + str(confidence))
                            continue
                    except Exception as e:
                        print("OBJECT DETECTOR ERROR: skipping frame..." + str(e))
                        self.frame_queue.task_done()
                        continue

                    #validates whether ROI was succesfully extracted before progressing
                    if roi_frames is None:
                        print("roi_frames is none....skipping frame")
                        self.frame_queue.task_done()
                        continue

                    print("ROI FRAMES: ", roi_frames.shape)
                    print("\nDetecting keypoints...")
                    try:
                        #Processess ROI through KD and returns an array of detected keypoints
                        coordinates = kd.process(roi_frames)
                        #outputs the results
                        print("\n{}\n".format(coordinates))
                        print(coordinates.shape)
                    except Exception as e:
                        print("KEYPOINT DETECTOR ERROR: skipping frame..." + str(e))
                    #marks processing as complete
                    self.frame_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    print("PROCESS FRAMES: Error has arisen due to:" +  str(e))
                    try:
                        # marks processing as complete even after errors
                        self.frame_queue.task_done()
                    except Exception as e:
                        print("Error arisen due to: " + str(e))
        except Exception as e:
            print("PROCESS THREAD: Error has arisen due to:" + str(e))



class RealtimePipeline:
    """Main class for running the realtime pipeline. Orchestrates the capture, display and processing of frames.
    This includes managing the created cpature and processing threads"""
    def __init__(self, process_every_n_frames=60):
        #Forces os's primary display (negates issues arising via ssh given commands)
        os.environ['DISPLAY'] = ':0'
        #TODO: Gstreamer pipeline. Elaborated in notion MAYBE add more context here later
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
        """Creates, configures and starts both threads befroe waiting for completion and cleanly shutting down"""
        try:
            #creates both thread instances
            self.capture_thread = FrameCaptureThread(self.gst_stream, self.frame_queue, self.stop_event, self.process_every_n_frames)
            self.process_thread = FrameProcessingThread(self.frame_queue, self.stop_event)

            #starts both threads
            self.capture_thread.start()
            self.process_thread.start()

            #waits for threads to complete
            self.capture_thread.join()
            self.process_thread.join()

        except KeyboardInterrupt:
            #Cleanly stops the threads based on user interruption
            print("Interrupted by Keyboard.")
            self.stop()
        except Exception as e:
            print("Error occurred due to: " + str(e))
        finally:
            self.stop()

    @staticmethod
    def run(process_every_n_frames):
        try:
            #creates and runs realtime pipeline instance
            pipeline = RealtimePipeline(process_every_n_frames)
            pipeline.process()
        except Exception as e:
            print("PIPELINE ERROR: occurred due to: " + str(e))




if __name__ == "__main__":
    # realtime_pipeline = RealtimePipeline()
    # realtime_pipeline.process()

    # An updated approach. Argparse approach means the number of runs can added to the cli command
    parser = argparse.ArgumentParser(description='Run a CV pipeline with camera capture and processing')
    parser.add_argument("--frames_interval", type=int, default=60, help="Process every N frmaes (60 default)")
    args = parser.parse_args()
    RealtimePipeline.run(process_every_n_frames=args.frames_interval)