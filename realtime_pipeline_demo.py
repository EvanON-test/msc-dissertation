#TODO: DONT FORGET TO cite the code sections you have used formally (gst, gfg etc)
#TODO: Replace print with logging info for better threading output
#TODO: test with 60, 30, 15 etc various levels for basic performance understanding
import queue
import time
import numpy as np
import cv2
import sys
import os
from threading import Thread, Event, Lock
import argparse
import datetime
import gc
import csv
import platform
import psutil
from queue import Queue
import tempfile



#Utilised try bocks to allow for failure, due to the wrong hardware
#jtop is a nano specific library for accessing hardware metrics
try:
    from jtop import jtop
except ImportError:
    jtop = None

import processing.binary_classifier_util as bc
import processing.frame_selector_util as fs
import processing.object_detector_util as od
import processing.keypoint_detector_util as kd



# #TODO: THIS IS DUPLICATED - CAN IMPROVE LATER ITERATIONS - Potentially slim it as just to save image & keypoints
class SaveDetectionThread(Thread):
    """A separate thread that further processes and saves information regarding the detections. Currently further processes the frame to get the keypoint data
    before saving it as a csv file as well as saving the frame as well as the frame with the bounded box on"""
    def __init__(self, frame, roi_frames, confidence, frame_counter):
        """Initialises the thread class as well as the detection data from the realtimepipeline that is needed for the further processing """
        super().__init__()
        self.frame = frame
        self.roi_frames = roi_frames
        self.confidence = confidence

        self.frame_counter = frame_counter
        self.output_directory = "./realtime_frames/"
        os.makedirs(self.output_directory, exist_ok=True)


    def run(self):
        """Main pipeline for processing and saving the data. Saves the frame as an image, modifies and saves an annotated copy
        and also processes via the KD and saves the output as a csv."""
        try:
            print("SAVE DETECTION THREAD: Loading Keypoint Detector...")
            kd.load_model()
        except Exception as e:
            print(f"SAVE DETECTION THREAD: Failed to load Keypoint Detector due to: {e}")
            return
        try:
            #cretes a timestamp used when creating unique directory and filenames
            creation_time = datetime.datetime.now()
            timestamp = creation_time.strftime("%Y-%m-%d_%H-%M-%S")

            #generates the unique directory name
            detection_dir = os.path.join(self.output_directory, f"{timestamp}_Detection")
            os.mkdir(detection_dir)


            # generates the unique filename for original image and saves it to the unique directory
            image_filename = f"{timestamp}_screenshot.jpg"
            path = os.path.join(detection_dir, image_filename)
            cv2.imwrite(path, self.frame)


            #processes the roi through the KD beofre returning the coordinates
            coordinates = kd.process(self.roi_frames)
            # generates the unique filename for keypoint information and flattens it beofre writing it ot hte csv file
            csv_filename = f"{timestamp}_keypoints.csv"
            csv_path = os.path.join(detection_dir, csv_filename)
            flattened_coordinates = coordinates.flatten()
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                #keypoint detector returns 7 keypoints with 2 cords each. Updated their names based on rereading the Research paper
                headers = ['crab_left_x1', 'crab_left_y1', 'crab_right_x2', 'crab_right_y2', 'left_eye_x3', 'left_eye_y3', 'right_eye_x4', 'right_eye_y4', 'carapace_end_x5', 'carapace_end_y5', 'tail_end_x6', 'tail_end_y6', 'last_segment_x7', 'last_segment_y7']
                writer.writerow(headers)
                writer.writerow(flattened_coordinates)
            print(f"SAVE DETECTION THREAD: Keypoints saved to: {csv_path}")


            #summary of saved files
            print(f"SAVE DETECTION THREAD: Saved to: {image_filename}")
            print(f"SAVE DETECTION THREAD: Detection confidence: {self.confidence:.2f}")
            kd.unload_model()
        except Exception as e:
            print(f"SAVE DETECTION THREAD: ERROR SAVING DETECTION...{e}")

#TODO: Iterate and improve on this threaded approach. USE MONITORING AND DOCS
class ObjectDetectorThread(Thread):
    def __init__(self, frame_queue, result_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            print("OD THREAD: Loading Object Detector...")
            od.load_model()
        except Exception as e:
            print(f"OD THREAD: Failed to load Object Detector due to: {e}")
            return
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=2)
                if frame_data is None:
                    continue

                frame, frame_counter = frame_data

                print(f"OD THREAD:Processing frame:  {frame_counter} for Object Detection")
                # processes frame through object detector which outputs region of interest and confidence level
                roi_frames, confidence = od.process_realtime(frame)
                print(f"OD THREAD: Frame processed successfully, confidence: {confidence:.2f}")
                self.result_queue.put((frame, roi_frames, confidence, frame_counter))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"OD THREAD: Error in Object Detection Thread: {e}")


class AnalysisThread(Thread):
    def __init__(self, analysis_queue, detection_queue):
        super().__init__()
        self.analysis_queue = analysis_queue
        self.detection_queue = detection_queue
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            print("ANALYSIS THREAD: Loading Binary Classifier and Frame Selector models...")
            bc.load_model()
            fs.load_model()
        except Exception as e:
            print(f"ANALYSIS THREAD: Failed to load Binary Classifier and Frame Selecto due to: {e}")
            return

        while self.running:
            try:
                frame_data = self.analysis_queue.get(timeout=2)
                if frame_data is None:
                    continue

                frames, start_frame = frame_data
                print(f"ANALYSIS THREAD: Processing frame: {len(frames)} from {start_frame} for Binary Classifier and Frame Selector")

                temp_video= tempfile.mktemp(suffix=".mp4")
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(temp_video, fourcc, 15.0, (width, height))

                for frame in frames:
                    video_writer.write(frame)

                video_writer.release()
                print("ANALYSIS THREAD: Temp Video created Successfully")

                try:
                    print("ANALYSIS THREAD: Attempting Binary Classification...")
                    capture = cv2.VideoCapture(temp_video)

                    if not capture.isOpened():
                        print("ANALYSIS THREAD: Failed to open video capture for BC")
                        continue

                    signal = bc.process_realtime(capture)
                    # signal = bc.process(capture)

                    capture.release()

                    print(f"ANALYSIS THREAD: Binary Classifier returned: {signal}")
                    print(f"ANALYSIS THREAD: Binary Classifier signal length: {len(signal)}")

                    positive_frames = sum(signal)

                    if positive_frames == 0:
                        print("ANALYSIS THREAD: No Crustacean detected - skipping FS processing")
                        continue

                    print("ANALYSIS THREAD: Attempting Frame Selection...")

                    capture = cv2.VideoCapture(temp_video)

                    if not capture.isOpened():
                        print("ANALYSIS THREAD: Failed to open video capture for BC")
                        continue

                    extracted_frame_idxs = fs.process_realtime(signal, capture)

                    capture.release()

                    print("ANALYSIS THREAD: EXTRACTED: ")
                    print(len(extracted_frame_idxs))
                    print(len(extracted_frame_idxs[0]))
                    print(len(extracted_frame_idxs[1]))

                    selected_index = None
                    if extracted_frame_idxs[0]:
                        selected_index = extracted_frame_idxs[0][0]
                    elif extracted_frame_idxs[1]:
                        selected_index = extracted_frame_idxs[1][0]

                    if selected_index is not None:
                        best_frame = frames[selected_index]
                        frame_number = start_frame + selected_index
                        print(f"ANALYSIS THREAD: Selected Frame: {frame_number} for Object Detection")

                        try:
                            self.detection_queue.put((best_frame.copy(), frame_number))
                            #TODO: add object detection here?
                        except Exception as e:
                            print(f"ANALYSIS THREAD: ERROR SAVING DETECTION...{e}")

                    else:
                        print("ANALYSIS THREAD: No good frame selected")

                finally:
                    try:
                        os.remove(temp_video)
                    except:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ANALYSIS THREAD: Error in Binary Classifier and Frame Selection Thread: {e}")







#TODO: update comments and README later (not changed since new approach)
class RealtimePipelineDemo:
    """Main class for running the realtime pipeline. Orchestrates the capture, display and processing of frames.
    This includes managing the created cpature and processing threads"""
    def __init__(self, process_every_n_frames=60):
        #Forces os's primary display (negates issues arising via ssh given commands)
        os.environ['DISPLAY'] = ':0'
        #TODO: Gstreamer pipeline. Elaborated in notion ADD more context here when cleaning up
        # self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720 ! nvvidconv ! videoflip method=rotate-180 ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e drop=true max-buffers=2 sync=false"
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720 ! nvvidconv ! videoflip method=rotate-180 ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false"
        self.process_every_n_frames = process_every_n_frames

        # self.detection_box = None
        self.detection_confidence = 0.0
        self.detection_age = 0

        #Stores previous frame for use in motion detection
        self.previous_frame = None
        #Minimum level, percentage, above which motion detection function is triggered
        self.detection_minimum = 15

        self.detection_count = 0
        # starttime used for calculating runtime
        self.start_time = 0

        #Metrics output
        self.jetson = jtop()

        #Object detection thread
        self.detection_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=6)
        self.detection_thread = ObjectDetectorThread(self.detection_queue, self.result_queue)

        self.analysis_queue = Queue(maxsize=2)
        self.analysis_thread = AnalysisThread(self.analysis_queue, self.detection_queue)

        self.collecting = False
        self.collected_frames = []
        self.collect_start = 0
        self.frames_needed = 30


    def get_metrics(self):
        """Gathers metrics that have common access approaches in both devices and the specific device"""
        metrics = {}
        try:
            metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            metrics['ram_percent'] = memory.percent
            # gets the nano metrics using the jtop service object
            metrics['cpu_temp'] = self.jetson.temperature.get('CPU').get('temp')
            metrics['gpu_temp'] = self.jetson.temperature.get('GPU').get('temp')
            # Power metrics not possible on this iteration of NVIDIA's device
            # metrics['power_used'] = "N/A"
        except Exception as e:
            print(f"Error getting metrics: {e}")
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
            print(f"REALTIME PIPELINE: Detected motion: {movement_percentage:.2f}%")
            return True
        else:
            print(f"REALTIME PIPELINE: Not significant motion: {movement_percentage:.2f}%")
            return False


    def run(self):
        """Creates, configures and starts both threads befroe waiting for completion and cleanly shutting down"""
        self.start_time = time.time()

        try:
            # Initialises camera capture utilising Gstreamer approach
            capture = cv2.VideoCapture(self.gst_stream, cv2.CAP_GSTREAMER)
            # Verifies camera opened succesfully
            if not capture.isOpened():
                print("REALTIME PIPELINE: GST Stream failed to open.")
                print(f"REALTIME PIPELINE: Pipeline: {self.gst_stream}")
                return

            # extracts the cameras properties
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

            frame_counter = 0
            fps_frame_counter = 0
            fps_timer_start = cv2.getTickCount()
            current_fps = 0.0

            print(f"REALTIME PIPELINE: Camera initialised successfully.")
            print(f"REALTIME PIPELINE: Processing every {self.process_every_n_frames} frames.")
            print(f"REALTIME PIPELINE: Press CTRL+C to exit or q to exit.")

            if self.jetson:
                self.jetson.start()

            self.detection_thread.start()
            self.analysis_thread.start()

            # The main loop, continues until quit
            while True:
                # reads next frame from camera
                ret, frame = capture.read()
                if not ret:
                    print("REALTIME PIPELINE: Failed to capture frame.")
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
                    try:
                        motion_detected = self.detect_motion(frame)
                    except Exception as e:
                        print(f"REALTIME PIPELINE: Error detecting motion: {e}")
                        motion_detected = False
                    if motion_detected and not self.collecting:
                        print("REALTIME PIPELINE: Motion Detected starting to collect")
                        self.collecting = True
                        self.collected_frames = []
                        self.collect_start = frame_counter

                if self.collecting:
                    self.collected_frames.append(frame.copy())
                    if len(self.collected_frames) >= self.frames_needed:
                        print("REALTIME PIPELINE: Collection complete")
                        try:
                            self.analysis_queue.put_nowait((self.collected_frames.copy(), self.collect_start))
                        except:
                            print("REALTIME PIPELINE: Analysis queue full")

                        self.collecting = False
                        self.collected_frames = []


                try:
                    while not self.result_queue.empty():
                        # frame, roi_frames, confidence, bbox, frame_counter = self.result_queue.get_nowait()
                        frame, roi_frames, confidence, frame_counter = self.result_queue.get_nowait()
                        print(f"REALTIME PIPELINE: Recieved detection result For frame:  {frame_counter}, Confidence: {confidence}")
                        if confidence > 0.75:
                            self.detection_age = 0
                            self.detection_confidence = confidence
                            # self.detection_box = bbox
                            print(f"REALTIME PIPELINE: Confidence sufficiently high: {confidence:.2f}")
                            try:
                                self.detection_count += 1
                                #Calling the save detection processes in another thread with all the detection data
                                saving_thread = SaveDetectionThread(frame.copy(), roi_frames, confidence, frame_counter)
                                saving_thread.start()
                            except Exception as e:
                                print(f'REALTIME PIPELINE: ERROR while implementing SaveDetectionThread: {e}')
                            # clean memory
                            del roi_frames, confidence
                            gc.collect()
                except Exception as e:
                    print(f"REALTIME PIPELINE: Detection Queue empty. Further Details: {e}")

                if self.detection_age < 25:
                    detection_text = f"REALTIME PIPELINE: Detection: {self.detection_confidence:.2f}"
                    cv2.putText(display_frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    self.detection_age += 1

                hardware_metrics = self.get_metrics()
                if frame_counter % self.process_every_n_frames == 0:
                    hardware_metrics = self.get_metrics()
                #builds an overlay string to be displayed
                display_info = f"Resolution: {width}x{height}, FPS: {current_fps}"
                # Adds text overlay to frame. Some is self explanatory. (10, 10) = (left, top). 0.5 = font size. (0, 255, 0) = hex colour green. 2 = text thickness
                cv2.putText(display_frame, display_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                cv2.putText(display_frame, f"CPU: {hardware_metrics['cpu_percent']}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"CPU Temp: {hardware_metrics['cpu_temp']} celsius", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"RAM: {hardware_metrics['ram_percent']}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"GPU: {hardware_metrics['gpu_temp']} celsius", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # displys the frame
                cv2.imshow('Live Feed', display_frame)

                # checks for user input. If q pressed initiates stop event
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("REALTIME PIPELINE: User pressed q to exit. Shutting down.")
                    break


        except KeyboardInterrupt:
            print("REALTIME PIPELINE: Interrupted by Keyboard.")
        except Exception as e:
            print(f"REALTIME PIPELINE: Error has arisen due to: {e}")
        finally:
            # Resouces minimisation after loops have completed
            capture.release()
            cv2.destroyAllWindows()
            od.unload_model()
            if self.jetson:
                self.jetson.close()
            if self.detection_thread.is_alive():
                self.detection_thread.stop()
                self.detection_thread.join()
            if self.analysis_thread.is_alive():
                self.analysis_thread.stop()
                self.analysis_thread.join()


            #Calulates the runtime and provides a summarisation of the overall run
            runtime = time.time() - self.start_time
            print("\n --- FINAL SUMMARY --- ")
            print(f"    High confidence detections saved: {self.detection_count}")
            print(f"    Total Runtime: {runtime}")




if __name__ == "__main__":
    # An updated approach. Argparse approach means the number of runs can added to the cli command
    parser = argparse.ArgumentParser(description='Run a CV pipeline with camera capture and processing')
    parser.add_argument("--frames_interval", type=int, default=60, help="Process every N frmaes (60 default)")
    args = parser.parse_args()
    realtime_pipeline = RealtimePipelineDemo(process_every_n_frames=args.frames_interval)
    realtime_pipeline.run()