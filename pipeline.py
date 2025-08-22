"""
The original Computer Vision Pipeline with minor improvements for readability and
usability.
"""

#Import statements for required modules
import shutil
import time
import cv2
import os
import argparse

#Import statements for custom model utils
import processing.binary_classifier_util as bc
import processing.frame_selector_util as fs
import processing.object_detector_util as od
import processing.keypoint_detector_util as kd



class Pipeline:
    """The main pipeline that orchestrates the processing workflow."""

    def __init__(self, completed_files_log="./CompletedFiles.txt", extracted_frames_dir="./processing/extracted_frames/" ):
        """Initialises the pipeline with easy modifiable file paths"""
        self.completed_files_log = completed_files_log
        self.extracted_frames_dir = extracted_frames_dir


    def run(self, data_path="processing/video", monitor=None, runs=1):
        """
        Main function to execute the pipeline. Handles two modes: normal mode  and a benchmarking mode.
        """

        try:
            with open(self.completed_files_log, 'r') as cfl:
                completed_files = [line.strip('\n') for line in cfl.readlines()]
        except FileNotFoundError:
            with open(self.completed_files_log, "x"):
                pass
            completed_files = []

        #NOTE: THIS ONLY WORKS AS INTENDED WHEN THERE IS A SINGULAR FILE PRESENT
        #BENCHMARKING MODE: A simple loop approach to process the file/s (singular currently) relevant to the user inputted run value.
        if runs > 1:
            run_count = 0
            while run_count < runs:
                for filename in os.listdir(data_path):
                    print(f"Processing {filename}. Run: {run_count}/{runs}")
                    self.process(data_path + '/' + filename, monitor)
                    print(f"\nProcessed {filename} for Run: {run_count}/{runs}\n")
                    run_count += 1
            print(f"\nFinished processing for a total of: {runs}\n")
        else:
            #NORMAL MODE: A simple loop approach that processes each file in the directory 1 time, that isn't already in completed files.
            # A record is then added to the completed files.
            for filename in os.listdir(data_path):
                if filename not in completed_files and '.mp4' in filename:
                    print(f"Processing {filename}")
                    #Checks if monitor instance present, runs appropriate option
                    self.process(data_path + '/' + filename)

                    # if monitor:
                    #     self.process(data_path + '/' + filename, monitor)
                    # else:
                    #     self.process(data_path+'/'+filename)
                    completed_files.append(filename)
                    #Logs the filename in the completed_files.txt file
                    with open(self.completed_files_log, 'a') as cfl:
                        cfl.write(filename+"\n")
                    print(f"Finished processing {filename}")
            print("\nProcessed all available Files!!\n")


    def process(self, video_path, monitor=None):
        """This is the core processing Pipeline"""

        #start timer for calculating final runtime
        start_time = time.time()

        ##BINARY CLASSIFICATION
        bc_start_time = time.time()

        print("\nLocating contigs...")
        #updates current stage value (if monitor instance running)
        if monitor:
            monitor.current_stage = "Binary Classifier"
        #BC processes the video and predictions are assigned to the signal
        signal = bc.process(cv2.VideoCapture(video_path))
        bc_time = time.time() - bc_start_time




        ##FRAME SELECTION
        fs_start_time = time.time()
        print("\nExtracting best frames...")
        #updates current stage value (if monitor instance running)
        if monitor:
            monitor.current_stage = "Frame Selector"
        #FS processes the predictions and extracts the best frames
        #best_frames[0] is top, best_frames[1] is bottom,
        extracted_frame_idxs = fs.process(
            signal, cv2.VideoCapture(video_path))
        del signal
        fs_time = time.time() - fs_start_time




        ##OBJECT DETECTOR
        od_start_time = time.time()
        # updates current stage value (if monitor instance running)
        if monitor:
            monitor.current_stage = "Object Detector"
        #make sure its empty (so that only frames from the current run are saved)
        savepoint = self.extracted_frames_dir #"./processing/extracted_frames/"
        shutil.rmtree(savepoint)
        os.mkdir(savepoint)

        video = cv2.VideoCapture(video_path)

        #prints total No. of sub arrays and number frames in both the top and bottom arrays
        print("EXTRACTED: ")
        print(len(extracted_frame_idxs))
        print(len(extracted_frame_idxs[0]))
        print(len(extracted_frame_idxs[1]))

        # take indices of top frames only. Extracts and saves the selected frames
        for i in range(len(extracted_frame_idxs[0])):
            #jumps to specific frame in video before saving as PNG
            video.set(cv2.CAP_PROP_POS_FRAMES, extracted_frame_idxs[0][i])
            success, image = video.read()
            # extracted_top_frames[i] = full_frames[extracted_frame_idxs[1][i]]
            cv2.imwrite("./processing/extracted_frames/{}.png".format(i), image)
        # delete frames np array
        del video
        del extracted_frame_idxs
        # can also delete raw video here

        print("\nCropping to region of interest...")
        #OD processes the videos in given directory and assigns the returned NumPy array of cropped frames to roi_frmaes
        roi_frames = od.process(savepoint)
        print("ROI FRAMES: ", roi_frames.shape)
        od_time = time.time() - od_start_time



        ##KEYPOINT DETECTOR
        kd_start_time = time.time()
        print("\nDetecting Keypoints...")
        # updates current stage value (if monitor instance running)
        if monitor:
            monitor.current_stage = "Keypoint Detector"
        #KD processes the NumPy array and assigns
        coordinates = kd.process(roi_frames)
        kd_time = time.time() - kd_start_time
        print("\n{}\n".format(coordinates))
        print(coordinates.shape)
        pipeline_time = time.time() - start_time

        #MODELS END



        print("\nFinished!\nPipeline took {} seconds to process \"{}\"".format(
            round(pipeline_time, 2), video_path))
        print(
            "\nBC: {:.4f}s ({:.2f}%)"
            "\nFS: {:.4f}s ({:.2f}%)"
            "\nOD: {:.4f}s ({:.2f}%)"
            "\nKD: {:.4f}s ({:.2f}%)\n".format(
            bc_time, (bc_time/pipeline_time)*100,
            fs_time, (fs_time/pipeline_time)*100,
            od_time, (od_time/pipeline_time)*100,
            kd_time, (kd_time/pipeline_time)*100))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a CV pipeline on saved video files')
    parser.add_argument("--data_path", type=str, default="processing/video" ,help="Path to folder holding video files")
    parser.add_argument("--runs", type=int, default=1 ,help="Number of runs to run the pipeline for") #Hangover from monitoring really but will keep for now
    args = parser.parse_args()
    pipeline = Pipeline()
    pipeline.run(data_path=args.data_path, runs=args.runs)