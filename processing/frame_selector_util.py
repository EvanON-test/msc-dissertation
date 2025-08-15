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

#NOTE: Configuration setup consistent with Binary Classifier

LOW_RES_WIDTH = 320
LOW_RES_HEIGHT = 180

preloaded_top_frame_selector = None
preloaded_bottom_frame_selector = None


#loads model using global variables
def load_model():
    global preloaded_top_frame_selector
    global preloaded_bottom_frame_selector
    print("Loading Frame Selector models...")
    try:
        # Loads the 'top' quality model
        top_frame_selector = {}
        top_frame_selector['model'] = tflite.Interpreter(model_path=
                                                         "./processing/frame_selector/top_con_norm_bal_mse_1000.tflite")
        top_frame_selector['model'].allocate_tensors()
        top_frame_selector['input_details'] = top_frame_selector[
            'model'].get_input_details()
        top_frame_selector['output_details'] = top_frame_selector[
            'model'].get_output_details()

        # Loads the 'bottom' quality model
        bottom_frame_selector = {}
        bottom_frame_selector['model'] = tflite.Interpreter(model_path=
                                                            "./processing/frame_selector/bottom_con_norm_bal_mse_1000.tflite")
        bottom_frame_selector['model'].allocate_tensors()
        bottom_frame_selector['input_details'] = bottom_frame_selector[
            'model'].get_input_details()
        bottom_frame_selector['output_details'] = bottom_frame_selector[
            'model'].get_output_details()


        preloaded_top_frame_selector = top_frame_selector
        preloaded_bottom_frame_selector = bottom_frame_selector
        print("Model loaded")
    except Exception as e:
        print("Preloading OD failed due to: " + str(e))

#Unloads model using global variables
def unload_model():
    global preloaded_top_frame_selector
    global preloaded_bottom_frame_selector
    print("Unloading Frame Selector model...")
    try:
        preloaded_top_frame_selector = None
        preloaded_bottom_frame_selector = None
        print("FS Model Unloaded")
    except Exception as e:
        print("Unloading FS failed due to: " + str(e))


#downscales the original image (1280x720) to the lower res values defined above
def rescale_image(image):
    return cv2.resize(image, (LOW_RES_WIDTH, LOW_RES_HEIGHT))


#reshapes into tf compatible format
def tf_reshape(image):
    X = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
    return X


def predict_quality(frame_selector, frame):
    """Assesses the quality of a single frame based on the model in the argument"""
    #sets input tensor
    frame_selector['model'].set_tensor(
        frame_selector['input_details'][0]['index'], 
        frame.astype(np.float32))
    #runs inference
    frame_selector['model'].invoke()
    #Assigns and returns quality prediction
    prediction = frame_selector['model'].get_tensor(
        frame_selector['output_details'][0]['index'])
    return prediction


def process(signal, video):
    """The main function of frame selection. Receives binary signal from classifier and
     identifies where continuous segments where prediction is present (1). For each segment evaluates using
      2 models and selects the single best frame and returns the indices of best frames"""

    #Loads the 'top' quality model
    top_frame_selector = {}
    top_frame_selector['model'] = tflite.Interpreter(model_path=
        "./processing/frame_selector/top_con_norm_bal_mse_1000.tflite")
    top_frame_selector['model'].allocate_tensors()
    top_frame_selector['input_details'] = top_frame_selector[
        'model'].get_input_details()
    top_frame_selector['output_details'] = top_frame_selector[
        'model'].get_output_details()

    # Loads the 'bottom' quality model
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

    # plot_t = np.zeros((total_frames))
    # plot_b = np.zeros((total_frames))

    #Arrays for qaulity scores for top nad bottom
    contig_t = []
    contig_b = []
    in_contig = False
    #Index of best frames for top and bottom
    current_best_t = None
    current_best_b = None
    #Array of best frame indices which is what is updated and returned at the end. Made up of top and bottom model selections
    best_frames = [[],[]]

    #Loops over all frames
    for i in range(total_frames):
        #Basically if i = 1 (crustacean present)
        if signal[i]:
            #flags as in contig (in a continuous segment)
            in_contig = True
            #pre-processes the frame (size, grey and shape)
            rescaled_frame = rescale_image(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            reshaped_frame = tf_reshape(rescaled_frame)
            #Assess the quality using both models, returning prediction to the variables
            t_quality = predict_quality(top_frame_selector, reshaped_frame)
            b_quality = predict_quality(bottom_frame_selector, reshaped_frame)

            #Updates the best frame index if currrent is best
            if len(contig_t) == 0 or t_quality > max(contig_t): current_best_t = i
            if len(contig_b) == 0 or b_quality > max(contig_b): current_best_b = i
            #stores the quality scores
            contig_t.append(t_quality)
            contig_b.append(b_quality)
        elif in_contig: #Was in segment but current frame has no crustacean (0)
            #marks segment as ended
            in_contig = False
            #saves the best frame for this segment
            best_frames[0].append(current_best_t)
            best_frames[1].append(current_best_b)
            #rests variables for next segment
            current_best_t = None
            current_best_b = None
            contig_t = []
            contig_b = []
        else: #Not in segment and no crustacean (0)
            in_contig = False
        #Next frame
        success, image = video.read()

    # unload models after use
    del top_frame_selector
    del bottom_frame_selector

    return best_frames # best_frames[0] is top, best_frames[1] is bottom, 

def process_realtime(signal, video):
    """The main function of frame selection. Receives binary signal from classifier and
     identifies where continuous segments where prediction is present (1). For each segment evaluates using
      2 models and selects the single best frame and returns the indices of best frames"""
    global preloaded_top_frame_selector
    global preloaded_bottom_frame_selector

    top_frame_selector = preloaded_top_frame_selector
    bottom_frame_selector = preloaded_bottom_frame_selector

    # linear search for highest quality frame
    # print("\nSelecting best frames...")

    # check loaded
    success, image = video.read()
    if not success:
        print("\nERROR\n")
        sys.exit()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # plot_t = np.zeros((total_frames))
    # plot_b = np.zeros((total_frames))

    #Arrays for qaulity scores for top nad bottom
    contig_t = []
    contig_b = []
    in_contig = False
    #Index of best frames for top and bottom
    current_best_t = None
    current_best_b = None
    #Array of best frame indices which is what is updated and returned at the end. Made up of top and bottom model selections
    best_frames = [[],[]]

    #Loops over all frames
    for i in range(total_frames):
        #Basically if i = 1 (crustacean present)
        if signal[i]:
            #flags as in contig (in a continuous segment)
            in_contig = True
            #pre-processes the frame (size, grey and shape)
            rescaled_frame = rescale_image(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            reshaped_frame = tf_reshape(rescaled_frame)
            #Assess the quality using both models, returning prediction to the variables
            t_quality = predict_quality(top_frame_selector, reshaped_frame)
            b_quality = predict_quality(bottom_frame_selector, reshaped_frame)

            #Updates the best frame index if currrent is best
            if len(contig_t) == 0 or t_quality > max(contig_t): current_best_t = i
            if len(contig_b) == 0 or b_quality > max(contig_b): current_best_b = i
            #stores the quality scores
            contig_t.append(t_quality)
            contig_b.append(b_quality)
        elif in_contig: #Was in segment but current frame has no crustacean (0)
            #marks segment as ended
            in_contig = False
            #saves the best frame for this segment
            best_frames[0].append(current_best_t)
            best_frames[1].append(current_best_b)
            #rests variables for next segment
            current_best_t = None
            current_best_b = None
            contig_t = []
            contig_b = []
        else: #Not in segment and no crustacean (0)
            in_contig = False
        #Next frame
        success, image = video.read()

    # # unload models after use
    # del top_frame_selector
    # del bottom_frame_selector

    return best_frames # best_frames[0] is top, best_frames[1] is bottom