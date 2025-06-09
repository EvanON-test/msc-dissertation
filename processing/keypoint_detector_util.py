import tflite_runtime.interpreter as tflite
import numpy as np
import json
import time
import cv2
import sys


def process(frames):

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=
        "./processing/keypoint_detector/models/32_4000_197.07_14.11.04.512680.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
    