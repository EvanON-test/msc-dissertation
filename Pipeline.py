import numpy as np
import shutil
import time
import cv2
import sys
import os

import processing.binary_classifier_util as bc
import processing.frame_selector_util as fs
import processing.object_detector_util as od
import processing.keypoint_detector_util as kd

COMPLETED_FILES_LOG = "./CompletedFiles.txt"

#Updated this into a class
class Pipeline:
    @staticmethod
    def run(data_path, monitor=None, runs=1):

        try:
            with open(COMPLETED_FILES_LOG, 'r') as cfl:
                completed_files = [line.strip('\n') for line in cfl.readlines()]
        except FileNotFoundError:
            f = open(COMPLETED_FILES_LOG, "x")
            completed_files = []

        """Implemented an approach to loop over the processing function based on the number of 'runs' input. Better data creation"""
        #TODO: this is only useful while there is 1 file in a benchmarking scenario - will need to adjust/remove at a later time
        if runs > 1:
            run_count = 0
            while run_count < runs:
                for filename in os.listdir(data_path):
                    print(f"Processing {filename}. Run: {run_count}/{runs}")
                    Pipeline.process(data_path + '/' + filename, monitor)
                    print(f"\nProcessed {filename} for Run: {run_count}/{runs}\n")
                    run_count += 1
            print(f"\nFinished processing for a total of: {runs}\n")
        else:
            for filename in os.listdir(data_path):
                if filename not in completed_files and '.mp4' in filename:
                    print(f"Processing {filename}")
                    #Checks if monitor instance, runs appropriate option
                    if monitor:
                        Pipeline.process(data_path + '/' + filename, monitor)
                    else:
                        Pipeline.process(data_path+'/'+filename)
                    completed_files.append(filename)
                    with open(COMPLETED_FILES_LOG, 'a') as cfl:
                        cfl.write(filename+"\n")
                    print(f"Finished processing {filename}")
            print("\nProcessed all available Files!!\n")


    @staticmethod
    def process(video_path, monitor=None):

        start_time = time.time()

        bc_start_time = time.time()

        print("\nLocating contigs...")
        #updates current stage value (if monitor instance running)
        if monitor:
            monitor.current_stage = "Binary Classifier"
        signal = bc.process(cv2.VideoCapture(video_path))

        bc_time = time.time() - bc_start_time

        fs_start_time = time.time()

        print("\nExtracting best frames...")
        if monitor:
            monitor.current_stage = "Frame Selector"
        extracted_frame_idxs = fs.process(
            signal, cv2.VideoCapture(video_path))
        del signal

        fs_time = time.time() - fs_start_time

        od_start_time = time.time()
        if monitor:
            monitor.current_stage = "Object Detector"
        # make sure its empty
        savepoint = "./processing/extracted_frames/"
        shutil.rmtree(savepoint)
        os.mkdir(savepoint)

        video = cv2.VideoCapture(video_path)

        print("EXTRACTED: ")
        print(len(extracted_frame_idxs))
        print(len(extracted_frame_idxs[0]))
        print(len(extracted_frame_idxs[1]))

        # take indices of top frames only
        for i in range(len(extracted_frame_idxs[0])):
            video.set(cv2.CAP_PROP_POS_FRAMES, extracted_frame_idxs[0][i])
            success, image = video.read()
            # extracted_top_frames[i] = full_frames[extracted_frame_idxs[1][i]]
            cv2.imwrite("./processing/extracted_frames/{}.png".format(i), image)
        # delete frames np array
        del video
        del extracted_frame_idxs
        # can also delete raw video here

        print("\nCropping to region of interest...")
        roi_frames = od.process(savepoint)

        print("ROI FRAMES: ", roi_frames.shape)

        od_time = time.time() - od_start_time

        kd_start_time = time.time()

        print("\nDetecting keypoints...")
        if monitor:
            monitor.current_stage = "Keypoint Detector"
        coordinates = kd.process(roi_frames)

        kd_time = time.time() - kd_start_time

        print("\n{}\n".format(coordinates))
        print(coordinates.shape)

        pipeline_time = time.time() - start_time
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
    #TODO: potentially introduce a args approach here for conformity (although not needed)
    pipeline = Pipeline()
    pipeline.run('processing/video')