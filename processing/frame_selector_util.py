# from frame_selector.single_image_cnn import SingleCNN
# import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import numpy as np
import random
import time
import copy
import sys
import cv2
import os


LOW_RES_WIDTH = 320
LOW_RES_HEIGHT = 180


def rescale_image(image):
    return cv2.resize(image, (LOW_RES_WIDTH, LOW_RES_HEIGHT))


# reshape in tf compatible format
def tf_reshape(image):
    X = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
    return X


def predict_quality(frame_selector, frame):
    frame_selector['model'].set_tensor(
        frame_selector['input_details'][0]['index'], 
        frame.astype(np.float32))
    frame_selector['model'].invoke()
    prediction = frame_selector['model'].get_tensor(
        frame_selector['output_details'][0]['index'])
    return prediction


def rectangle_smooth(preds, gamma):
    smoothed = np.zeros((preds.shape[0]))
    for i in range(len(preds)):
        window = np.concatenate((
            preds[max(0, i-(gamma//2)):i],
            preds[i:min(preds.shape[0], i+(gamma//2))]))
        smoothed[i] = np.mean(window)
    return smoothed


def process(signal, video):

    top_frame_selector = {}
    top_frame_selector['model'] = tflite.Interpreter(model_path=
        "./processing/frame_selector/top_con_norm_bal_mse_1000.tflite")
    top_frame_selector['model'].allocate_tensors()
    top_frame_selector['input_details'] = top_frame_selector[
        'model'].get_input_details()
    top_frame_selector['output_details'] = top_frame_selector[
        'model'].get_output_details()

    bottom_frame_selector = {}
    bottom_frame_selector['model'] = tflite.Interpreter(model_path=
        "./processing/frame_selector/bottom_con_norm_bal_mse_1000.tflite")
    bottom_frame_selector['model'].allocate_tensors()
    bottom_frame_selector['input_details'] = bottom_frame_selector[
        'model'].get_input_details()
    bottom_frame_selector['output_details'] = bottom_frame_selector[
        'model'].get_output_details()

    # linear search for highest quality frame
    # print("\nSelecting best frames...")

    # check loaded
    success, image = video.read()
    if not success:
        print("\nERROR\n")
        sys.exit()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    plot_t = np.zeros((total_frames))
    plot_b = np.zeros((total_frames))

    contig_t = []
    contig_b = []
    in_contig = False
    current_best_t = None
    current_best_b = None
    best_frames = [[],[]]

    for i in range(total_frames):
        if signal[i]:
            in_contig = True
            rescaled_frame = rescale_image(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            reshaped_frame = tf_reshape(rescaled_frame)
            t_quality = predict_quality(top_frame_selector, reshaped_frame)
            b_quality = predict_quality(bottom_frame_selector, reshaped_frame)

            if len(contig_t) == 0 or t_quality > max(contig_t): current_best_t = i
            if len(contig_b) == 0 or b_quality > max(contig_b): current_best_b = i
            contig_t.append(t_quality)
            contig_b.append(b_quality)
        elif in_contig:
            in_contig = False
            best_frames[0].append(current_best_t)
            best_frames[1].append(current_best_b)
            current_best_t = None
            current_best_b = None
            contig_t = []
            contig_b = []
        else:
            in_contig = False

        success, image = video.read()

    # unload models after use
    del top_frame_selector
    del bottom_frame_selector

    return best_frames # best_frames[0] is top, best_frames[1] is bottom, 
