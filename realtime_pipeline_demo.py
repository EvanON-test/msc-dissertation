#TODO: DONT FORGET TO cite the code sections you have used formally (gst, gfg etc)
#TODO: update comments and README later (not changed since new approach)
# TODO: Elaborate on Gstreamer pipeline in report. Elaborated in notion ADD more context here when cleaning up
"""
The updated real-time implementation of the original Computer Vision Pipeline

Multi-threaded approach constituting of:

    - MainThread: Capture from camera, motion detection, display and orchestration
    - AnalysisThread: Performs binary classification and frame selection
    - ObjectDetectorThread: Processes frames selected for object detection
    - SaveDetectionThread:   Processes keypoint detection and saved detection results
"""

#TODO: FUTURE JOB - Iterate realtime_pipeline_demo and realtime_pipeline as most code is duplicated
#Import statements for required modules
import queue
import time
import cv2
import os
from threading import Thread
import argparse
import datetime
import gc
import csv
import psutil
from queue import Queue
import tempfile

#jtop is a nano specific library for accessing hardware metrics
try:
    from jtop import jtop
except ImportError:
    jtop = None

#Import statements for custom model utils
import processing.binary_classifier_util as bc
import processing.frame_selector_util as fs
import processing.object_detector_util as od
import processing.keypoint_detector_util as kd




class SaveDetectionThread(Thread):
    """A seperate thread that further processes and saves information regarding high confidence detections.

        - Loads keypoint detector
        - generates a unique directory
        - generates a unique filename and saves the frame as a jpg
        - processes roi frames through keypoint detector and saves results as a csv
        - unloads keypoint detector model
    """
    def __init__(self, frame, roi_frames, confidence, frame_counter):
        """Initialises the thread as well as the detection data given as arguments when thread object is first created, from the realtimepipeline """
        super().__init__()
        self.frame = frame
        self.roi_frames = roi_frames
        self.confidence = confidence
        self.frame_counter = frame_counter

        #sets output directory and creates if it doesnt exist
        self.output_directory = "./realtime_frames/"
        os.makedirs(self.output_directory, exist_ok=True)


    def run(self):
        """Main pipeline for SaveDetection Thread. Saves the frame as an image, processes roi via the keypoint detector and saves the output as a csv."""
        # Loads the keypoint detection model and stops and returns an informative error message if it fails
        try:
            print("SAVE DETECTION THREAD: Loading Keypoint Detector...")
            kd.load_model()
        except Exception as e:
            print(f"SAVE DETECTION THREAD: Failed to load Keypoint Detector due to: {e}")
            return
        try:
            # cretes a timestamp used when creating unique directory and filenames
            creation_time = datetime.datetime.now()
            timestamp = creation_time.strftime("%Y-%m-%d_%H-%M-%S")

            # generates the unique directory name
            detection_dir = os.path.join(self.output_directory, f"{timestamp}_Detection")
            os.mkdir(detection_dir)


            # generates the unique filename for original frmae and saves it to the unique directory
            image_filename = f"{timestamp}_screenshot.jpg"
            path = os.path.join(detection_dir, image_filename)
            cv2.imwrite(path, self.frame)


            # processes the roi through the KD beofre returning the coordinates
            coordinates = kd.realtime_process(self.roi_frames)
            # generates the unique filename for keypoint information and flattens it beofre writing it ot hte csv file
            csv_filename = f"{timestamp}_keypoints.csv"
            csv_path = os.path.join(detection_dir, csv_filename)
            #flattens the co-ordinate array for csv writing. keypoint detector returns 7 keypoints with 2 cords each.
            flattened_coordinates = coordinates.flatten()
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                #TODO: CITE PAPER?
                #I updated their names to align with the details in the foundational paper
                headers = ['crab_left_x1', 'crab_left_y1',
                           'crab_right_x2', 'crab_right_y2',
                           'left_eye_x3', 'left_eye_y3',
                           'right_eye_x4', 'right_eye_y4',
                           'carapace_end_x5', 'carapace_end_y5',
                           'tail_end_x6', 'tail_end_y6',
                           'last_segment_x7', 'last_segment_y7']
                #writes headers row follow by the keypoints
                writer.writerow(headers)
                writer.writerow(flattened_coordinates)
            print(f"SAVE DETECTION THREAD: Keypoints saved to: {csv_path}")


            # prints a summary of saved files
            print(f"SAVE DETECTION THREAD: Saved to: {image_filename}")
            print(f"SAVE DETECTION THREAD: Detection confidence: {self.confidence:.2f}")
            #Unloads model to save resources
            kd.unload_model()
        except Exception as e:
            print(f"SAVE DETECTION THREAD: ERROR SAVING DETECTION...{e}")




class ObjectDetectorThread(Thread):
    """A seperate thread that processes frames through object detection.

            - Loads object detector
            - receives a frame and frame number from analysis thread's queue
            - processes frame through object detector and assigns returned roi and confidence values
            - puts frame, roi info, confidence and frame number into result queue to be returned to Main thread
        """

    def __init__(self, detection_queue, result_queue):
        """Initialises Thread and assigns queues """
        super().__init__()
        self.detection_queue = detection_queue
        self.result_queue = result_queue
        self.running = True

    def stop(self):
        """Assigns the stop flag for the thread"""
        self.running = False

    def run(self):
        """Main processing loop for object detection. Processes frames from the input detection queue and placing results in
        the result output queue
        """
        # Loads the object detection model, stops and returns an informative error message if it fails
        try:
            print("OD THREAD: Loading Object Detector...")
            od.load_model()
        except Exception as e:
            print(f"OD THREAD: Failed to load Object Detector due to: {e}")
            return

        #Main processing loop. Continues until stop called (and running = False)
        while self.running:
            try:
                #gets next frame from detection queue (with a timeout for periodic checks)
                frame_data = self.detection_queue.get(timeout=2)
                if frame_data is None:
                    continue

                #Assigns frame and frame number (from queue) to variables
                frame, frame_counter = frame_data

                print(f"OD THREAD:Processing frame:  {frame_counter} for Object Detection")
                # processes frame through object detector which outputs region of interest and confidence level
                roi_frames, confidence = od.process_realtime(frame)
                print(f"OD THREAD: Frame processed successfully, confidence: {confidence:.2f}")
                #Adds frame, roi_frames, confidence and frame counter into results queue
                self.result_queue.put((frame, roi_frames, confidence, frame_counter))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"OD THREAD: Error in Object Detection Thread: {e}")


class AnalysisThread(Thread):
    """A seperate thread that processes frames through binary classification and frame selection.

     This includes:

         - Loads binary classifier and frame selector models
         - receives a frame and frame number from main threads analysis queue
         - creates a temporary video file and writes frames to it (bc util expects videofile)
         - processes temporary video file through binary classifier, assigns details to signal
         - processes temporary video file and signal through frame selector, assigns extracted best frames
         - Selects optimal frame for object detection
         - Puts selected frame in object detection queue
     """

    def __init__(self, analysis_queue, detection_queue):
        """Initialises Thread and assigns queues """
        super().__init__()
        self.analysis_queue = analysis_queue
        self.detection_queue = detection_queue
        self.running = True

    def stop(self):
        """Defines the stop flag for the thread"""
        self.running = False

    def run(self):
        try:
            #Loads both binary classifier and frame selector models
            print("ANALYSIS THREAD: Loading Binary Classifier and Frame Selector models...")
            bc.load_model()
            fs.load_model()
        except Exception as e:
            print(f"ANALYSIS THREAD: Failed to load Binary Classifier and Frame Selecto due to: {e}")
            return
        #Main processing loop that will run until stop called
        while self.running:
            try:
                # gets next frame sequence, 30 frame array,  from analysis queue (with a timeout for periodic checks)
                frame_data = self.analysis_queue.get(timeout=2)
                if frame_data is None:
                    continue

                #Assigns frame array and frame number (from queue) to variables
                frames, start_frame = frame_data
                print(f"ANALYSIS THREAD: Processing frames: {len(frames)} from start point {start_frame} for Binary Classifier and Frame Selector")

                #TODO: CITE
                #Creates temporary vido file (model utils require them for processing)
                temp_video= tempfile.mktemp(suffix=".mp4")
                height, width = frames[0].shape[:2]
                #configures video writer with mp4 codec (to align with all other videos in project files)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(temp_video, fourcc, 15.0, (width, height))

                #Writes all the frames to temporary video
                for frame in frames:
                    video_writer.write(frame)
                video_writer.release()
                print("ANALYSIS THREAD: Temp Video created Successfully")


                try:
                    #BINARY CLASSIFICATION
                    print("ANALYSIS THREAD: Attempting Binary Classification...")

                    capture = cv2.VideoCapture(temp_video)
                    if not capture.isOpened():
                        print("ANALYSIS THREAD: Failed to open video capture for BC")
                        continue
                    #Assigns signal array indicating presence per frame of animal
                    # signal = bc.process(capture)
                    signal = bc.process_realtime(capture)
                    capture.release()
                    #prints entire array and displays presence, 1, or non presence, 0.
                    print(f"ANALYSIS THREAD: Binary Classifier returned: {signal}")
                    print(f"ANALYSIS THREAD: Binary Classifier signal length: {len(signal)}")

                    #Verifies the presence of positive values (animal presence)
                    positive_frames = sum(signal)
                    if positive_frames == 0:
                        print("ANALYSIS THREAD: No Crustacean detected - skipping FS processing")
                        continue

                    print("ANALYSIS THREAD: Attempting Frame Selection...")
                    capture = cv2.VideoCapture(temp_video)
                    if not capture.isOpened():
                        print("ANALYSIS THREAD: Failed to open video capture for BC")
                        continue

                    # FS processes the predictions and extracts the best frames
                    # best_frames[0] is top, best_frames[1] is bottom,
                    extracted_frame_idxs = fs.process_realtime(signal, capture)
                    # extracted_frame_idxs = fs.process(signal, capture)
                    capture.release()

                    # prints total No. of sub arrays and number frames in both the top and bottom arrays
                    print("ANALYSIS THREAD: EXTRACTED: ")
                    print(len(extracted_frame_idxs))
                    print(len(extracted_frame_idxs[0]))
                    print(len(extracted_frame_idxs[1]))

                    #Selects top frame with a fallback to bottom frame
                    selected_index = None
                    if extracted_frame_idxs[0]:
                        selected_index = extracted_frame_idxs[0][0]
                    elif extracted_frame_idxs[1]:
                        selected_index = extracted_frame_idxs[1][0]

                    # Checks if selected index not none and selects the best frame and its number
                    if selected_index is not None:
                        best_frame = frames[selected_index]
                        frame_number = start_frame + selected_index
                        print(f"ANALYSIS THREAD: Selected Frame: {frame_number} for Object Detection")
                        #attmpts to add the variables into the detection queue for use in object detection thread
                        try:
                            self.detection_queue.put((best_frame.copy(), frame_number))
                        except Exception as e:
                            print(f"ANALYSIS THREAD: ERROR SAVING DETECTION...{e}")
                    else:
                        print("ANALYSIS THREAD: No good frame selected")
                finally:
                    #cleans up memory regardless of outcome
                    try:
                        os.remove(temp_video)
                    except:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ANALYSIS THREAD: Error in Binary Classifier and Frame Selection Thread: {e}")





class RealtimePipelineDemo:
    """This is the main orchestrator class for the entire realtime pipeline.

    This includes:

        - Camera capture and display
        - motion detection and triggering events
        - frame collection and analysis
        - coordination of the multi-threaded approach
        - collection of hardware metrics
        - User interface display
    """
    def __init__(self, process_every_n_frames=30):
        """Initialises Thread and assigns queues. Assigns frame process cadence as 30 by default """
        #Forces os's primary display (negates issues arising due to ssh commands)
        os.environ['DISPLAY'] = ':0'
        #TODO: CITE
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720, framerate=45/1 ! nvvidconv ! videoflip method=rotate-180 ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false"
        self.process_every_n_frames = process_every_n_frames

        #most recent detection confidence
        self.detection_confidence = 0.0
        #total count of confident detections
        self.detection_count = 0
        # starttime used for calculating runtime
        self.start_time = 0

        #Stores previous frame for use in calculating changes between frames for motion detection
        self.previous_frame = None
        #Minimum level, in %, above which motion detection function is triggered
        self.detection_minimum = 15

        #jtop to be interogated for metrics
        self.jetson = jtop()

        #TODO: CITE
        #THREADING FOCUS
        #FLOW REMINDER: MAIN -> ANALYSIS -> OBJECTDETECTION -> MAIN -> SAVEDETECTION
        self.analysis_queue = Queue(maxsize=1) #Frame sequence sent to ANALYSIS
        self.detection_queue = Queue(maxsize=1) #Selected frame from ANALYSIS to OBJECT DETECTION
        self.result_queue = Queue(maxsize=1) #Detection results sent from OBJECT DETECTION to MAIN

        #Initialises seperate threads for frame processing. ANALYSIS (bc & fs) and DETECTION (od)
        self.analysis_thread = AnalysisThread(self.analysis_queue, self.detection_queue)
        self.detection_thread = ObjectDetectorThread(self.detection_queue, self.result_queue)

        #variables for motion detection management
        self.collecting = False
        self.collected_frames = [] #accumulated frames for
        self.collect_start = 0 #allows for the recording of the frame at which collection began
        self.frames_needed = 30 #total frames required

        #variables to action detection cooldown
        self.last_detection_time = 0
        self.detection_cooldown = 3




    def get_metrics(self):
        """Gathers hardware metrics to display to user for realtime performance monitoring"""
        #initilises empty dictionary
        metrics = {}
        try:
            # Adds the current CPU utilisation percentage as a value to cpu_percent key
            metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            # Adds the current RAM utilisation percentage as a value to ram_percent key
            memory = psutil.virtual_memory()
            metrics['ram_percent'] = memory.percent
            # gets the nano metrics using the jtop service object
            # and adds the current cpu temp and gpu temp, as celsius, to the cpu_temp and gpu_temp keys respectively
            metrics['cpu_temp'] = self.jetson.temperature.get('CPU').get('temp')
            metrics['gpu_temp'] = self.jetson.temperature.get('GPU').get('temp')
            # Power metrics not possible on this iteration of NVIDIA's device
            # metrics['power_used'] = "N/A"
        except Exception as e:
            print(f"Error getting metrics: {e}")
        return metrics




    #TODO: CITE
    def detect_motion(self, frame):
        """Detects motion between consecutive frames by calculating the pixel difference between the current frame and the previous """
        #converts the frame to greyscale
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Intialises previous frame to current frame, for first call
        if self.previous_frame is None:
            self.previous_frame = grey_image
            return False

        #clacualtes absolute difference between the frames
        frame_diff = cv2.absdiff(self.previous_frame, grey_image)
        #updates the reference frame for next iteration
        self.previous_frame = grey_image

        #Applies threshold to highlight significant differences
        #Pixels iwth a difference greater than detection minimum become white, others black
        _, thresh = cv2.threshold(frame_diff, self.detection_minimum, 255, cv2.THRESH_BINARY)

        #Calcualtes the percentage of the pixels that changed significantly
        non_zero_count = cv2.countNonZero(thresh)
        total_pixels = thresh.size
        movement_percentage = (non_zero_count / total_pixels) * 100
       #Checks if the percentage greater than minimum and returns boolean
        if movement_percentage > self.detection_minimum:
            print(f"REALTIME PIPELINE: Detected motion: {movement_percentage:.2f}%")
            return True
        else:
            print(f"REALTIME PIPELINE: Not significant motion: {movement_percentage:.2f}%")
            return False


    #TODO: carry on from here - 1-2 hours with util's i would guess

    def run(self):
        """The main method that executes the orchestrating logic of the entire realtime pipeline

        Main Pipeline Flow:

            - Initialises camera


        """
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

                    current_time = time.time()
                    time_since_last_detection = current_time - self.last_detection_time

                    if motion_detected and not self.collecting and time_since_last_detection> self.detection_cooldown:
                        print("REALTIME PIPELINE: Motion Detected starting to collect")
                        self.collecting = True
                        self.collected_frames = []
                        self.collect_start = frame_counter
                        self.last_detection_time = current_time
                        print(f"REALTIME PIPELINE: Starting Motion detection cooldown period of {self.detection_cooldown} seconds.")
                    elif motion_detected and time_since_last_detection <= self.detection_cooldown:
                        print("REALTIME PIPELINE: In Motion detection cooldown period.")

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
                            # self.detection_age = 0
                            self.detection_confidence = confidence
                            print(f"REALTIME PIPELINE: Confidence sufficiently high: {confidence:.2f}")
                            cv2.putText(display_frame, f"Crustacean Detection Confidence: {confidence:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
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


                status = f"REALTIME PIPELINE: Collecting frames {len(self.collected_frames)}/{self.frames_needed}"
                cv2.putText(display_frame, status, (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                if not self.analysis_queue.empty():
                    cv2.putText(display_frame, "ANALYZING FRAMES...", (440, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                if not self.detection_queue.empty():
                    cv2.putText(display_frame, "DETECTING OBJECT", (460, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.putText(display_frame, f"DETECTION COUNT: {self.detection_count}", (480, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                hardware_metrics = self.get_metrics()
                if frame_counter % self.process_every_n_frames == 0:
                    hardware_metrics = self.get_metrics()
                #builds an overlay string to be displayed
                display_info = f"Resolution: {width}x{height}, FPS: {current_fps}"
                # Adds text overlay to frame. Some is self explanatory. (10, 10) = (left, top). 0.5 = font size. (0, 255, 0) = hex colour green. 2 = text thickness
                cv2.putText(display_frame, display_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                cv2.putText(display_frame, f"CPU: {hardware_metrics['cpu_percent']}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"RAM: {hardware_metrics['ram_percent']}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"CPU Temp: {hardware_metrics['cpu_temp']} celsius", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
    parser.add_argument("--frames_interval", type=int, default=30, help="Process every N frmaes (30 default)")
    args = parser.parse_args()
    realtime_pipeline = RealtimePipelineDemo(process_every_n_frames=args.frames_interval)
    realtime_pipeline.run()