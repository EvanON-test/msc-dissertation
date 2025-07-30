#TODO: update comments and README later (not changed since new approach)
import csv
import numpy as np
import cv2
import sys
import os
from threading import Thread
import argparse
import datetime
import gc
import time

import processing.object_detector_util as od
import processing.keypoint_detector_util as kd

class SaveDetectionThread(Thread):
    """A separate thread that further processes and saves information regarding the detections. Currently further processes the frame to get the keypoint data
    before saving it as a csv file as well as saving the frame as well as the frame with the bounded box on (TODO: move bounding box to demo when it works)"""
    def __init__(self, frame, roi_frames, confidence, bbox, frame_counter):
        """Initialises the thread class as well as the detection data from the realtimepipeline that is needed for the further processing """
        super().__init__()
        self.frame = frame
        self.roi_frames = roi_frames
        self.confidence = confidence
        self.bbox = bbox
        self.frame_counter = frame_counter
        self.output_directory = "./realtime_frames/"
        os.makedirs(self.output_directory, exist_ok=True)


    def run(self):
        """Main pipeline for processing and saving the data. Saves the frame as an image, modifies and saves an annotated copy
        and also processes via the KD and saves the output as a csv."""
        try:
            print("Loading Keypoint Detector...")
            kd.load_model()
        except Exception as e:
            print(f"Failed to load Keypoint Detector due to: {e}")
            return
        try:
            #cretes a timestamp used when creating unique directory and filenames
            creation_time = datetime.datetime.now()
            timestamp = creation_time.strftime("%Y-%m-%d_%H-%M-%S")

            #generates the unique directory name
            detection_dir = os.path.join(self.output_directory, f"{timestamp}_Detection")
            os.mkdir(detection_dir)

            #TODO: remove this after debugging finished
            frame_with_bbox = self.frame.copy()
            x1, y1, x2, y2 = self.bbox
            frame_height, frame_width = frame_with_bbox.shape[:2]
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            print(f"Frame width: {frame_width}, Frame height: {frame_height}")
            print(f"Bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"Bbox width: {bbox_width}, Bbox height: {bbox_height}")

            #TODO: fix this after
            #Draws a green box around the detected object (that is the aim at least)
            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #Adds the info alongside the bounding box
            detection_text = f"Detection: {self.confidence:.2f}"
            cv2.putText(frame_with_bbox, detection_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # generates the unique filename for original image and saves it to the unique directory
            image_filename = f"{timestamp}_screenshot.jpg"
            path = os.path.join(detection_dir, image_filename)
            cv2.imwrite(path, self.frame)

            #TODO debug by saving and checking roi_frmaes
            #generates the unique filename for original image and saves it to the unique directory
            roi_image_filename = f"roi_{timestamp}_screenshot.jpg"
            path = os.path.join(detection_dir, roi_image_filename)
            cv2.imwrite(path, self.roi_frames)

            # generates the unique filename for annotated image and saves it to the unique directory
            bbox_image_filename = f"bbox_{timestamp}_screenshot.jpg"
            bbox_path = os.path.join(detection_dir, bbox_image_filename)
            cv2.imwrite(bbox_path, frame_with_bbox)
            print(f"Saved high confidence frame: {self.confidence:.2f}")

            #processes the roi through the KD beofre returning the coordinates
            coordinates = kd.process(self.roi_frames)
            # generates the unique filename for keypoint information and flattens it beofre writing it ot hte csv file
            csv_filename = f"{timestamp}_keypoints.csv"
            csv_path = os.path.join(detection_dir, csv_filename)
            flattened_coordinates = coordinates.flatten()
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                #keypoint detector returns 7 keypoints with 2 cords each. Updated their names based on rereading the Research paper
                # headers = ['x1, y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7']
                headers = ['crab_left', 'crab_right', 'left_eye', 'right_eye', 'carapace_end', 'tail_end']
                writer.writerow(headers)
                writer.writerow(flattened_coordinates)
            print(f"Keypoints saved to: {csv_path}")


            #summary of saved files
            print(f"Saved to: {image_filename}")
            print(f"Detection confidence: {self.confidence:.2f}")
            print(f"Bounding Box: {self.bbox}")
        except Exception as e:
            print(f"ERROR SAVING DETECTION...{e}")

#TODO: can find a way to test your new pipeline on the saved video?
class RealtimePipeline:
    """Main class for running the realtime pipeline. Orchestrates the capture, motion detection and processing of frames
     before saving high confidence detections, in a separate thread."""
    def __init__(self, process_every_n_frames=30):
        #TODO: Gstreamer pipeline. Elaborated in notion MAYBE add more context later
        #TODO: test different resolutions (640, 480 original)
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=15/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e"
        #cadence of frames to process
        self.process_every_n_frames = process_every_n_frames
        # Threshold at which above it  will engage detection to be saved
        self.confidence_threshold = 0.85

        #Stores most recent detection
        self.detection_box = None
        #Stores latest confidence level
        self.detection_confidence = 0.0
        #Stores the count of high confidence detections
        self.detection_count = 0
        #starttime used for calculating runtime
        self.start_time = 0

        #Stores previous frame for use in motion detection
        self.previous_frame = None
        #Minimum level, percentage, above which motion detection function is triggered
        self.detection_minimum = 20

    #TODO: DEF REF
    def detect_motion(self, frame):
        """Detects motion between consecutive frames by comparing the current frame to the previous """
        #converts the frame to greyscale
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Intialises peprevious frame to current frame, for first call
        if self.previous_frame is None:
            self.previous_frame = grey_image
            return False

        #clacualtes absolute difference between the frames
        frame_diff = cv2.absdiff(self.previous_frame, grey_image)
        self.previous_frame = grey_image

        #Applies threshold to highlight significant differences
        #Pixels iwth a difference greater than detection minimum (currently 30) become white, others black
        _, thresh = cv2.threshold(frame_diff, self.detection_minimum, 255, cv2.THRESH_BINARY)

        #Calcualtes the percentage of the pixels that changed significantly
        non_zero_count = cv2.countNonZero(thresh)
        total_pixels = thresh.size
        movement_percentage = (non_zero_count / total_pixels) * 100
       #Checks if the percentage greater than minimum (currently 30) and returns boolean
        if movement_percentage > self.detection_minimum:
            print(f"Detected motion: {movement_percentage:.2f}%")
            return True
        else:
            print(f"Not significant motion: {movement_percentage:.2f}%")
            return False




    def run(self):
        #Records starttime for runtime calculations
        start_time = time.time()
        #Loads Object detector model
        od.load_model()
        try:
            # Initialises camera capture utilising Gstreamer approach
            capture = cv2.VideoCapture(self.gst_stream, cv2.CAP_GSTREAMER)
            # Verifies camera opened succesfully
            if capture.isOpened() == False:
                print("GST Stream failed to open.")
                return
            #Initialises variable to count total frames captured
            frame_counter = 0

            #Outputs summary preceding main capture & processing loop
            print(f"\nCAMERA INITIALISED SUCCESSFULLY.")
            print(f"PROCESSING EVERY {self.process_every_n_frames} FRAMES.")
            print(f"Press CTRL+C to exit or q to exit.")

            #Main loop, continues until quit
            while True:
                # Reads next frame from camera
                ret, frame = capture.read()
                #checks the capture frame has failed, skips this iteration of the loop if so
                if not ret:
                    print("Failed to capture frame.")
                    continue

                #clacualtes whether the current frame count is a multiple of predefined processing cadence (currently 30)
                #If it is then checks for motion (based on function defined earlier)
                frame_counter += 1
                if frame_counter % self.process_every_n_frames == 0:
                    motion_detected = self.detect_motion(frame)

                    #TODO: Need to adjust this. Not good as is. Decrease n frames?
                    #Checsk if motion deteceted, skips this iteration of the loop if no motion
                    if not motion_detected:
                        print("Failed to detect motion.")
                        continue
                    try:
                        print(f"Processing frame:  {frame_counter} for Object Detection")
                        # processes frame through object detector which outputs region of interest, confidence level and bounding box
                        roi_frames, confidence, bbox = od.process_realtime(frame)
                        # print(f"Frame processed successfully, confidence: {confidence:.2f}")
                        if confidence > self.confidence_threshold:
                            print(f"Confidence sufficiently high: {confidence:.2f}")
                            try:
                                #Increments the detection count before calling the save detection processes in another thread with all the detection data
                                self.detection_count += 1
                                saving_thread = SaveDetectionThread(frame.copy(), roi_frames, confidence, bbox, frame_counter)
                                saving_thread.start()
                            except Exception as e:
                                print(f'ERROR while implementing SaveDetectionThread: {e}')
                            ##Hard codes a wait into the pipeline in a, poor, attempt to minimise multiple counts of the same crustacean
                            #TODO: THIS IS A VERY POOR APPROACH. ITERATE.....ALTHOUGH IT DOES SEEM TO HELP
                            wait_time = 4
                            print(f"WAITING: Waiting for {wait_time} seconds to prevent duplicates.\n")
                            time.sleep(wait_time)
                        else:
                            print(f"Confidence below threshold\n")

                        #Helps minimise memory usage by deleting variables and garbage cleaning
                        del roi_frames, confidence, bbox
                        gc.collect()
                    except Exception as e:
                        print(f"OD PROCESSING ERROR. Caused by: {e}")
        except KeyboardInterrupt:
            print("Interrupted by Keyboard.")
        except Exception as e:
            print(f"CAPTRUE THREAD: Error has arisen due to: {e}")
        finally:
            #Cleans up resources in all eventualities
            print("Shutting down Resources...")
            capture.release()
            od.unload_model()
            kd.unload_model()
            del roi_frames, confidence, bbox
            gc.collect()

            #Calulates the runtime and provides a summarisation of the overall run
            runtime = time.time() - start_time
            print("\n --- FINAL SUMMARY --- ")
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