import tflite_runtime.interpreter as tflite
import numpy as np
import json
import time
import cv2
import sys


preloaded_interpreter = None
preloaded_input_details = None
preloaded_output_details = None

#loads model using global variables
def load_model():
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details
    print("KEYPOINT UTIL: Loading Keypoint Detector model...")
    try:
        preloaded_interpreter = tflite.Interpreter(model_path=
        "./processing/keypoint_detector/models/32_4000_197.07_14.11.04.512680.tflite")
        preloaded_interpreter.allocate_tensors()
        preloaded_input_details = preloaded_interpreter.get_input_details()
        preloaded_output_details = preloaded_interpreter.get_output_details()
        print("KEYPOINT UTIL: Model loaded")
    except Exception as e:
        print("KEYPOINT UTIL: Preloading KD failed due to: " + str(e))

#Unloads model using global variables
def unload_model():
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details
    print("KEYPOINT UTIL: Unloading Keypoint Detector model...")
    try:
        preloaded_interpreter = None
        preloaded_input_details = None
        preloaded_output_details = None
        print("KEYPOINT UTIL: Model Unloaded")
    except Exception as e:
        print("KEYPOINT UTIL: Unloading KD failed due to: " + str(e))


def process(frames):
    """Main detection function. Receives cropped ROI's and detects the keypoint co-ordinates within each
     before returning as a flattened array"""

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=
        "./processing/keypoint_detector/models/32_4000_197.07_14.11.04.512680.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #pre-initialises output array for the keypoint co-ords
    coords = np.zeros((frames.shape[0], 14))

    #for loop processes each frame through the keypoint detection model
    for i in range(len(frames)):

        # frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # width = 114
        # height = 80
        # frame = cv2.resize(frame, (height, width))

        #reshaped for model input
        input_data = np.reshape(
            frames[i], (1, frames[i].shape[0], frames[i].shape[1], 1))

        #sets input datatype
        interpreter.set_tensor(
            input_details[0]['index'], input_data.astype(np.float32))

        #executes the model
        interpreter.invoke()

        #Assigns output tensor to index in coords array
        coords[i] = interpreter.get_tensor(output_details[0]['index'])

    return coords

def realtime_process(frames):
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details

    # Load TFLite model and allocate tensors.
    interpreter = preloaded_interpreter

    # Get input and output tensors.
    input_details = preloaded_input_details
    output_details = preloaded_output_details

    coords = np.zeros((frames.shape[0], 14))
    for i in range(len(frames)):

        # frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # width = 114
        # height = 80
        # frame = cv2.resize(frame, (height, width))

        input_data = np.reshape(
            frames[i], (1, frames[i].shape[0], frames[i].shape[1], 1))

        interpreter.set_tensor(
            input_details[0]['index'], input_data.astype(np.float32))

        interpreter.invoke()

        coords[i] = interpreter.get_tensor(output_details[0]['index'])

    return coords