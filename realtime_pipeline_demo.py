#TODO: DONT FORGET TO cite the code sections you have used formally (gst, gfg etc)
#TODO: test with 60, 30, 15 etc various levels for basic performance understanding
import time

import numpy as np
import cv2
import sys
import os
from threading import Thread, Event, Lock
import argparse
import datetime
import pipeline
import gc
import csv
import platform
import psutil

#Utilised try bocks to allow for failure, due to the wrong hardware
#jtop is a nano specific library for accessing hardware metrics
try:
    from jtop import jtop
except ImportError:
    jtop = None
#gpiozero provides CPU temperature on the Pi's.
try:
    from gpiozero import CPUTemperature
except ImportError:
    CPUTemperature = None

import processing.object_detector_util as od
import processing.keypoint_detector_util as kd



# #TODO: THIS IS DUPLICATED - CLEAN UOP LATER THOUGH - Potentially slim it as just to save image & keypoints
class SaveDetectionThread(Thread):
    """A separate thread that further processes and saves information regarding the detections. Currently further processes the frame to get the keypoint data
    before saving it as a csv file as well as saving the frame as well as the frame with the bounded box on"""
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
            # frame_height, frame_width = frame_with_bbox.shape[:2]
            # bbox_width = x2 - x1
            # bbox_height = y2 - y1
            # print(f"Frame width: {frame_width}, Frame height: {frame_height}")
            # print(f"Bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            # print(f"Bbox width: {bbox_width}, Bbox height: {bbox_height}")

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
                # TODO: add 7 more headers and do an x and y version of each
                #keypoint detector returns 7 keypoints with 2 cords each. Updated their names based on rereading the Research paper
                # headers = ['x1, y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7']
                headers = ['crab_left', 'crab_right', 'left_eye', 'right_eye', 'carapace_end', 'tail_end', 'last_segment']
                writer.writerow(headers)
                writer.writerow(flattened_coordinates)
            print(f"Keypoints saved to: {csv_path}")


            #summary of saved files
            print(f"Saved to: {image_filename}")
            print(f"Detection confidence: {self.confidence:.2f}")
            print(f"Bounding Box: {self.bbox}")
        except Exception as e:
            print(f"ERROR SAVING DETECTION...{e}")




#TODO: update comments and README later (not changed since new approach)
#TODO: adjusted back to 30 frmaes
class RealtimePipelineDemo:
    """Main class for running the realtime pipeline. Orchestrates the capture, display and processing of frames.
    This includes managing the created cpature and processing threads"""
    def __init__(self, process_every_n_frames=30):
        #Forces os's primary display (negates issues arising via ssh given commands)
        os.environ['DISPLAY'] = ':0'
        #TODO: Gstreamer pipeline. Elaborated in notion ADD more context here when cleaning up
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=10/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e"
        self.process_every_n_frames = process_every_n_frames

        self.detection_box = None
        self.detection_confidence = 0.0
        self.detection_age = 0

        #Stores previous frame for use in motion detection
        self.previous_frame = None
        #Minimum level, percentage, above which motion detection function is triggered
        self.detection_minimum = 20

        self.detection_count = 0
        # starttime used for calculating runtime
        self.start_time = 0

        #Metrics output
        self.jetson = jtop()

    def get_metrics(self):
        """Gathers metrics that have common access approaches in both devices and the specific device"""
        machine = platform.machine()
        metrics = {}
        metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        metrics['ram_percent'] = memory.percent
        if machine == "armv7l":
            cpu_temp_pi = CPUTemperature().temperature
            metrics['cpu_temp'] = round(cpu_temp_pi, 1)
            # # TODO: Reinvestigate the options for these later
            # # Sets the currently non-gatherable metrics to None
            # metrics['gpu_percent'] = "N/A"
            # metrics['gpu_temp'] = "N/A"
            # metrics['power_used'] = "N/A"
        elif machine == "aarch64":
            # gets the nano metrics using the jtop service object
            metrics['cpu_temp'] = self.jetson.temperature.get('CPU').get('temp')
            metrics['gpu_temp'] = self.jetson.temperature.get('GPU').get('temp')
            # Power metrics not possible on this iteration of NVIDIA's device
            # metrics['power_used'] = "N/A"
        else:
            print("Unknown machine")
        return metrics



    #TODO: refrence this whole block
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
        """Creates, configures and starts both threads befroe waiting for completion and cleanly shutting down"""
        self.start_time = time.time()
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

            # extracts the cameras properties
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

            frame_counter = 0
            fps_frame_counter = 0
            fps_timer_start = cv2.getTickCount()
            current_fps = 0.0

            print(f"Camera initialised successfully.")
            print(f"Processing every {self.process_every_n_frames} frames.")
            print(f"Press CTRL+C to exit or q to exit.")

            if self.jetson:
                self.jetson.start()

            # The main loop, continues until quit
            while True:
                # reads next frame from camera
                ret, frame = capture.read()
                if not ret:
                    print("Failed to capture frame.")
                    continue

                display_frame = frame.copy()

                fps_frame_counter += 1
                if fps_frame_counter >= 30:
                    fps_timer_end = cv2.getTickCount()
                    elapsed_time = (fps_timer_end - fps_timer_start) / cv2.getTickFrequency()
                    current_fps = fps_frame_counter / elapsed_time
                    fps_timer_start = cv2.getTickCount()
                    fps_frame_counter = 0

                frame_counter += 1

                if frame_counter % self.process_every_n_frames == 0:
                    hardware_metrics = self.get_metrics()
                    motion_detected = self.detect_motion(frame)
                    if not motion_detected:
                        print("Failed to detect motion.")
                        continue
                    else:
                        try:
                            print(f"Processing frame:  {frame_counter} for Object Detection")
                            # processes frame through object detector which outputs region of interest and confidence level
                            roi_frames, confidence, bbox = od.process_realtime(frame)
                            print(f"Frame processed successfully, confidence: {confidence:.2f}")
                            if confidence > 0.70:
                                self.detection_age = 0
                                self.detection_confidence = confidence
                                self.detection_box = bbox
                                print(f"Confidence sufficiently high: {confidence:.2f}")
                                try:
                                    self.detection_count += 1
                                    #Calling the save detection processes in another thread with all the detection data
                                    saving_thread = SaveDetectionThread(frame.copy(), roi_frames, confidence, bbox, frame_counter)
                                    saving_thread.start()

                                    #TODO: could always reintroduce a wait time here while testing
                                except Exception as e:
                                    print(f'ERROR while implementing SaveDetectionThread: {e}')
                                # clean memory
                                del roi_frames, confidence, bbox
                                gc.collect()
                        except Exception as e:
                            print(f"OD ERROR. Caused by: {e}")
                if self.detection_age < 15:
                    detection_text = f"Detection: {self.detection_confidence:.2f}"
                    cv2.putText(display_frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if self.detection_box is not None:
                        #bounding box
                        x1, y1, x2, y2 = self.detection_box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    self.detection_age += 1

                #builds an overlay string to be displayed
                display_info = f"Resolution: {width}x{height}, FPS: {current_fps}"
                hardware_info = f"CPU Percent: {hardware_metrics['cpu_percent']}%, CPU Temp: {hardware_metrics['cpu_temp']}, RAM Percent:{hardware_metrics['ram_percent']} GPU: {hardware_metrics['gpu_temp']}"
                # Adds text overlay to frame. Some is self explanatory. (10, 10) = (left, top). 0.5 = font size. (0, 255, 0) = hex colour green. 2 = text thickness
                cv2.putText(display_frame, display_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, hardware_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # displys the frame
                cv2.imshow('Live Feed', display_frame)

                # checks for user input. If q pressed initiates stop event
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User pressed q to exit. Shutting down.")
                    break


        except KeyboardInterrupt:
            print("Interrupted by Keyboard.")


        except Exception as e:
            print(f"CAPTRUE THREAD: Error has arisen due to: {e}")
        finally:
            # Resouces minimisation after loops have completed
            capture.release()
            cv2.destroyAllWindows()
            od.unload_model()
            if self.jetson:
                self.jetson.close()
            # kd.unload_model()

            #Calulates the runtime and provides a summarisation of the overall run
            runtime = time.time() - self.start_time
            print("\n --- FINAL SUMMARY --- ")
            print(f"    High confidence detections saved: {self.detection_count}")
            print(f"    Total Runtime: {runtime}")




if __name__ == "__main__":
    # An updated approach. Argparse approach means the number of runs can added to the cli command
    parser = argparse.ArgumentParser(description='Run a CV pipeline with camera capture and processing')
    parser.add_argument("--frames_interval", type=int, default=30, help="Process every N frmaes (30 default)")
    args = parser.parse_args()
    realtime_pipeline = RealtimePipelineDemo(process_every_n_frames=args.frames_interval)
    realtime_pipeline.run()