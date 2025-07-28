import numpy as np
import cv2
import sys
import os
from threading import Thread, Event, Lock
import argparse
import datetime
import gc
import time

import processing.object_detector_util as od
import processing.keypoint_detector_util as kd




#TODO: update comments and README later (not changed since new approach)
class RealtimePipeline:
    """Main class for running the realtime pipeline. Orchestrates the capture, display and processing of frames."""
    def __init__(self, process_every_n_frames=30):
        #TODO: Gstreamer pipeline. Elaborated in notion MAYBE add more context later
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=15/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e"
        self.process_every_n_frames = process_every_n_frames
        self.confidence_threshold = 0.70
        self.output_directory = "./realtime_frames/"
        self.detection_box = None
        self.detection_confidence = 0.0
        self.detection_count = 0
        self.start_time = 0

        os.makedirs(self.output_directory, exist_ok=True)

    # def process_keypoints(self, roi_frames):
    #     try:
    #         coordinates = kd.process(roi_frames)
    #         # outputs the results
    #         print("\n{}\n".format(coordinates))
    #         print(coordinates.shape)
    #         return coordinates
    #     except Exception as e:
    #         print("KEYPOINT DETECTION ERROR: skipping frame..." + str(e))

    def save_detection(self, frame, roi_frames, confidence, bbox, frame_counter):
        try:
            creation_time = datetime.datetime.now()
            timestamp = creation_time.strftime("%Y-%m-%d_%H-%M-%S")
            image_filename = f"{timestamp}_frame_{frame_counter}confidence_{confidence:.2f}.jpg"
            path = os.path.join(self.output_directory,image_filename)
            cv2.imwrite(path, frame)
            print(f"Saved high confidence frame: {confidence:.2f}")

            self.detection_count += 1

            print(f"Detection count ({self.detection_count}): {image_filename}")
            print(f"Detection confidence: {confidence:.2f}")
            print(f"Bounding Box: {bbox}")
        except Exception as e:
            print("ERROR SAVING DETECTION..." + str(e))



    def run(self):
        start_time = time.time()
        #Load the models
        od.load_model()
        # kd.load_model()
        try:
            # Initialises camera capture utilising Gstreamer approach
            capture = cv2.VideoCapture(self.gst_stream, cv2.CAP_GSTREAMER)
            # Verifies camera opened succesfully
            if capture.isOpened() == False:
                print("GST Stream failed to open.")
                return

            frame_counter = 0



            print(f"CAMERA INITIALISED SUCCESSFULLY.")
            print(f"PROCESSING EVERY {self.process_every_n_frames} FRAMES.")
            print(f"Press CTRL+C to exit or q to exit.")

            # The main loop, continues until quit
            while True:
                # reads next frame from camera
                ret, frame = capture.read()
                if not ret:
                    print("Failed to capture frame.")
                    continue

                frame_counter += 1
                if frame_counter % self.process_every_n_frames == 0:
                    try:
                        print(f"Processing frame:  {frame_counter} for Object Detection")
                        # processes frame through object detector which outputs region of interest and confidence level
                        roi_frames, confidence, bbox = od.process_realtime(frame)
                        print(f"Frame processed successfully, confidence: {confidence:.2f}")
                        if confidence > self.confidence_threshold:
                            print(f"Confidence sufficiently high: {confidence:.2f}")
                            self.save_detection(frame, roi_frames, confidence, bbox, frame_counter)
                            #TODO: THIS IS A VERY POOR APPROACH. ITERATE.....POTENTIALLY
                            wait_time = 3
                            print(f"WAITING: Waiting for {wait_time} seconds to prevent duplicates.")
                            time.sleep(wait_time)
                        else:
                            print(f"Confidence below threshold")

                        #clean memory
                        del roi_frames
                        gc.collect()
                    except Exception as e:
                        print(f"OD PROCESSING ERROR. Caused by: {e}")

                if frame_counter % 300 == 0:
                    print(f"Status Update: {frame_counter} frames processed, {self.detection_count} detections saved.")

        except KeyboardInterrupt:
            print("Interrupted by Keyboard.")
        except Exception as e:
            print(f"CAPTRUE THREAD: Error has arisen due to: {e}")
        finally:
            # Resouces minimisation after loops have completed
            print("Shutting down Resources...")
            capture.release()
            od.unload_model()
            # kd.unload_model()
            runtime = time.time() - start_time
            print(" --- FINAL SUMMARY --- ")
            print(f"    Total Frames Captured: {frame_counter}")
            print(f"    Frames Processed for Detection: {frame_counter // self.process_every_n_frames}")
            print(f"    High confidence detections saved: {self.detection_count}")
            print(f"    Total Runtime: {runtime}")




if __name__ == "__main__":
    # An updated approach. Argparse approach means the number of runs can added to the cli command
    parser = argparse.ArgumentParser(description='Run a CV pipeline with camera capture and processing')
    parser.add_argument("--frames_interval", type=int, default=30, help="Process every N frmaes (30 default)")
    args = parser.parse_args()
    realtime_pipeline = RealtimePipeline(process_every_n_frames=args.frames_interval)
    realtime_pipeline.run()