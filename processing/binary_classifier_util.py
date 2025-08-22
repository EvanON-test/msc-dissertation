# from binary_classifier.model.mobilenet_v3_small import MobileNetV3_Small
# import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import numpy as np
import random
import time
import json
import copy
import math
import sys
import cv2
import os

LOW_RES_WIDTH = 320
LOW_RES_HEIGHT = 180

BATCH_SIZE = 1

preloaded_interpreter = None
preloaded_input_details = None
preloaded_output_details = None

#loads model using global variables
def load_model():
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details
    print("BC UTIL: Loading Binary Classifier model...")
    try:
        # print("\nLoading binary_classifier...")
        model = {}
        # Loads tensorflowlite model
        model['classifier'] = tflite.Interpreter(
            model_path="./processing/binary_classifier/save/DS1_A_200_128.tflite")
        # configrues the input tensor shape
        model['classifier'].resize_tensor_input(
            0, [BATCH_SIZE, LOW_RES_HEIGHT, LOW_RES_WIDTH, 1])
        model['classifier'].allocate_tensors()

        # Get input and output tensors.
        model['input_details'] = model['classifier'].get_input_details()
        model['output_details'] = model['classifier'].get_output_details()

        preloaded_interpreter = model
        preloaded_input_details = model['input_details']
        preloaded_output_details = model['output_details']
        print("BC UTIL: BC Model loaded")
    except Exception as e:
        print("BC UTIL: Preloading BC failed due to: " + str(e))

#Unloads model using global variables
def unload_model():
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details
    print("BC UTIL: Unloading Binary Classifier model...")
    try:
        preloaded_interpreter = None
        preloaded_input_details = None
        preloaded_output_details = None
        print("BC Model Unloaded")
    except Exception as e:
        print("BC UTIL: Unloading BC failed due to: " + str(e))


#downscales the original image (1280x720) to the lower res values defined above
def rescale_image(image):
    return cv2.resize(image, (LOW_RES_WIDTH, LOW_RES_HEIGHT))


#reshapes into tf compatible format
def tensorflow_reshape(batch):
    tf_batch = np.reshape(batch, 
        (BATCH_SIZE, batch.shape[1], batch.shape[2], 1))
    return tf_batch


def classify_video(video, model):
    """The Main Classification function. Reads frames from the video before pre-processing each frame (by converting to grayscale and downscaling size)
    and running through inference before extracting and saving predictions"""
    # check loaded
    success, image = video.read()
    if not success:
        print("\nBC UTIL: ERROR\n")
        sys.exit()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    preds = np.zeros((total_frames))

    #processes the video in batches (1 currently)
    for i in range(0, total_frames, BATCH_SIZE):
        # ignore channels as images will be grayscale
        batch = np.zeros((BATCH_SIZE, LOW_RES_HEIGHT, LOW_RES_WIDTH))
        for b in range(0, BATCH_SIZE):
            if success:
                # grayscale and rescale
                batch[b] = rescale_image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                success, image = video.read()

        #reshapes into tf compatible format
        tf_batch = tensorflow_reshape(batch)
        #sets input tensor with the pre-processed data
        model['classifier'].set_tensor(
            model['input_details'][0]['index'], 
            tf_batch.astype(np.float32))
        #runs inference
        model['classifier'].invoke()
        #extracts predictions
        prediction = model['classifier'].get_tensor(
            model['output_details'][0]['index'])
        #stores prediction in array
        for p in range(prediction.shape[0]):
            try:
                preds[i+p] = prediction[p]
            except IndexError:
                pass # remainder of batch is padded zeros

    return preds


# essentially a convolution that overwrites
#    each prediction value with a moving average.
#    window is size gamma, centered on the ith pred.
def rectangle_smooth(preds, gamma):
    smoothed = np.zeros((preds.shape[0]))
    for i in range(len(preds)):
        window = np.concatenate((
            preds[max(0, i-(gamma//2)):i],
            preds[i:min(preds.shape[0], i+(gamma//2))]))
        smoothed[i] = np.mean(window)
    return smoothed


# Step function that snaps everything
#     to zero or one based on threshold
def rectify(sn, theta):
    smooth = copy.deepcopy(sn)
    for i in range(smooth.shape[0]):
        if smooth[i] < theta:
            smooth[i] = 0
        else:
            smooth[i] = 1
    return smooth




def process(video):
    """Main process function, called from the pipeline. Loads the model, processes all frames
    before applying smoothing and thresholding and returning a binary value indicating presence"""
    # print("\nLoading binary_classifier...")
    model = {}
    #Loads tensorflowlite model
    model['classifier'] = tflite.Interpreter(
        model_path="./processing/binary_classifier/save/DS1_A_200_128.tflite")
    #configrues the input tensor shape
    model['classifier'].resize_tensor_input(
        0, [BATCH_SIZE, LOW_RES_HEIGHT, LOW_RES_WIDTH, 1])
    model['classifier'].allocate_tensors()

    # Get input and output tensors.
    model['input_details'] = model['classifier'].get_input_details()
    model['output_details'] = model['classifier'].get_output_details()

    # print("\nPredicting...")
    #processses video and gets predictions
    preds = classify_video(video, model)


    # binary classifier is noisy
    #     so applying smoothing functions
    s1 = rectangle_smooth(preds, 20)
    r1 = rectify(s1, 0.01)
    s2 = rectangle_smooth(r1, 10)
    r2 = rectify(s2, 0.5)

    # unload model after use
    del model

    return r2



def process_realtime(video):
    """Main process function, called from the pipeline. Loads the model, processes all frames
    before applying smoothing and thresholding and returning a binary value indicating presence"""
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details

    # modififed interpreter use utilising preloaded model
    model = preloaded_interpreter

    # print("\nPredicting...")
    #processses video and gets predictions
    preds = classify_video(video, model)


    # binary classifier is noisy
    #     so applying smoothing functions
    s1 = rectangle_smooth(preds, 20)
    r1 = rectify(s1, 0.01)
    s2 = rectangle_smooth(r1, 10)
    r2 = rectify(s2, 0.5)

    # # unload model after use
    # del model


    return r2