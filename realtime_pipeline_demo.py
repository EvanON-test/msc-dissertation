#TODO: DONT FORGET TO cite the code sections you have used formally (gst, gfg etc)
#TODO: test with 60, 30, 15 etc various levels for basic performance understanding

import numpy as np
import cv2
import sys
import os
from threading import Thread, Event, Lock
import argparse
import datetime
import pipeline
import gc

import processing.object_detector_util as od
import processing.keypoint_detector_util as kd



# #TODO: ADD A SEPERATE THREAD for KD after completing the non video version
# class KeypointDetection(Thread):
#
#     # if roi_frames is None:
#     #     print("roi_frames is none....skipping frame")
#     #     self.frame_queue.task_done()
#     #     continue
#     def
#     print("ROI FRAMES: ", roi_frames.shape)
#     print("\nDetecting keypoints...")
#     try:
#         # Processess ROI through KD and returns an array of detected keypoints
#         coordinates = kd.process(roi_frames)
#         # outputs the results
#         print("\n{}\n".format(coordinates))
#         print(coordinates.shape)
#     except Exception as e:
#         print("KEYPOINT DETECTOR ERROR: skipping frame..." + str(e))
#     # marks processing as complete
#     self.frame_queue.task_done()

# #TODO: ADD A SEPERATE THREAD for OD after completing the non video version
# class ObjectDetection(Thread):


#TODO: re-build the code  into a singluar thread again - ONLY OD - KD AFTER TEST
#TODO: update comments and README later (not changed since new approach)
class RealtimePipelineDemo:
    """Main class for running the realtime pipeline. Orchestrates the capture, display and processing of frames.
    This includes managing the created cpature and processing threads"""
    def __init__(self, process_every_n_frames=15):
        #Forces os's primary display (negates issues arising via ssh given commands)
        os.environ['DISPLAY'] = ':0'
        #TODO: Gstreamer pipeline. Elaborated in notion MAYBE add more context here later
        self.gst_stream = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink -e"
        self.process_every_n_frames = process_every_n_frames


        self.detection_box = None
        self.detection_confidence = 0.0
        self.detection_age = 0


    def run(self):
        """Creates, configures and starts both threads befroe waiting for completion and cleanly shutting down"""

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
                    try:
                        print(f"Processing frame:  {frame_counter} for Object Detection")
                        # processes frame through object detector which outputs region of interest and confidence level
                        roi_frames, confidence, bbox = od.process_realtime(frame)
                        print(f"Frame processed successfully, confidence: {confidence:.2f}")
                        #TODO: add the bbox element after test
                        if confidence > 0.70:
                            self.detection_age = 0
                            self.detection_confidence = confidence
                            self.detection_box = bbox
                            #TODO: add this to the KD/secondary thread
                            print(f"Confidence sufficiently high: {confidence:.2f}")
                            # output_directory = "./realtime_frames/"
                            # os.makedirs(output_directory, exist_ok=True)
                            # # intilises a unique time based and confidence based name for the file
                            # creation_time = datetime.datetime.now()
                            # timestamp = creation_time.strftime("%Y-%m-%d_%H-%M")
                            # filename = os.path.join(output_directory, f"{timestamp}_confidence_{confidence:.2f}.jpg")
                            # cv2.imwrite(filename, frame)
                            # print(f"Saved high confidence frame: {confidence:.2f}")

                        #clean memory
                        del roi_frames
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

                # builds an overlay string to be displayed
                display_info = f"Resolution: {width}x{height}, FPS: {current_fps}"
                # Adds text overlay to frame. Some is self explanatory. (10, 10) = (left, top). 0.5 = font size. (0, 255, 0) = hex colour green. 2 = text thickness
                cv2.putText(display_frame, display_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
            # kd.unload_model()




if __name__ == "__main__":
    # An updated approach. Argparse approach means the number of runs can added to the cli command
    parser = argparse.ArgumentParser(description='Run a CV pipeline with camera capture and processing')
    parser.add_argument("--frames_interval", type=int, default=60, help="Process every N frmaes (60 default)")
    args = parser.parse_args()
    realtime_pipeline = RealtimePipelineDemo(process_every_n_frames=args.frames_interval)
    realtime_pipeline.run()