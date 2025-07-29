import csv
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

class SaveDetectionThread(Thread):
    def __init__(self, frame, roi_frames, confidence, bbox, frame_counter):
        super().__init__()
        self.frame = frame
        self.roi_frames = roi_frames
        self.confidence = confidence
        self.bbox = bbox
        self.frame_counter = frame_counter
        self.output_directory = "./realtime_frames/"
        os.makedirs(self.output_directory, exist_ok=True)


    def run(self):
        try:
            print("Loading Keypoint Detector...")
            kd.load_model()
        except Exception as e:
            print(f"Failed to load Keypoint Detector due to: {e}")
            return
        #TODO: test this approach
        try:
            creation_time = datetime.datetime.now()
            timestamp = creation_time.strftime("%Y-%m-%d_%H-%M-%S")

            detection_dir = os.path.join(self.output_directory, f"{timestamp}_Detection")
            os.mkdir(detection_dir)

            frame_with_bbox = self.frame.copy()
            x1, y1, x2, y2 = self.bbox

            frame_height, frame_width = frame_with_bbox.shape[:2]
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            #DEBUG
            print(f"Frame width: {frame_width}, Frame height: {frame_height}")
            print(f"Bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"Bbox width: {bbox_width}, Bbox height: {bbox_height}")

            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

            detection_text = f"Detection: {self.confidence:.2f}"
            cv2.putText(frame_with_bbox, detection_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            image_filename = f"{timestamp}_screenshot.jpg"
            path = os.path.join(detection_dir, image_filename)
            cv2.imwrite(path, self.frame)

            bbox_image_filename = f"bbox_{timestamp}_screenshot.jpg"
            bbox_path = os.path.join(detection_dir, bbox_image_filename)
            cv2.imwrite(bbox_path, frame_with_bbox)
            print(f"Saved high confidence frame: {self.confidence:.2f}")

            coordinates = kd.process(self.roi_frames)
            csv_filename = f"{timestamp}_keypoints.csv"
            csv_path = os.path.join(detection_dir, csv_filename)
            flattened_coordinates = coordinates.flatten()
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                #keypoint detector returns 7 keypoints with 2 cords each
                headers = ['x1, y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7']
                writer.writerow(headers)
                writer.writerow(flattened_coordinates)
            print(f"Keypoints saved to: {csv_path}")



            print(f"Saved to: {image_filename}")
            print(f"Detection confidence: {self.confidence:.2f}")
            print(f"Bounding Box: {self.bbox}")
        except Exception as e:
            print(f"ERROR SAVING DETECTION...{e}")

#TODO: update comments and README later (not changed since new approach)
class RealtimePipeline:
    """Main class for running the realtime pipeline. Orchestrates the capture, display and processing of frames."""
    def __init__(self, process_every_n_frames=30):
        #TODO: Gstreamer pipeline. Elaborated in notion MAYBE add more context later
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=15/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e"
        self.process_every_n_frames = process_every_n_frames
        self.confidence_threshold = 0.85
        self.detection_box = None
        self.detection_confidence = 0.0
        self.detection_count = 0
        self.start_time = 0





    def run(self):
        start_time = time.time()
        #Load the models
        od.load_model()
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
                            try:
                                self.detection_count += 1
                                saving_thread = SaveDetectionThread(frame.copy(), roi_frames, confidence, bbox, frame_counter)
                                saving_thread.start()
                            except Exception as e:
                                print(f'ERROR while implementing SaveDetectionThread: {e}')
                            #TODO: THIS IS A VERY POOR APPROACH. ITERATE.....ALTHOUGH IT DOES SEEM TO WORK
                            wait_time = 3
                            print(f"WAITING: Waiting for {wait_time} seconds to prevent duplicates.\n")
                            time.sleep(wait_time)
                        else:
                            print(f"Confidence below threshold\n")

                        #clean memory
                        del roi_frames, confidence, bbox
                        gc.collect()
                    except Exception as e:
                        print(f"OD PROCESSING ERROR. Caused by: {e}")



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