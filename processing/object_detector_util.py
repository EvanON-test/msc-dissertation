from processing.object_detector.nms import non_max_suppression
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import time
import cv2
import sys
import os

preloaded_interpreter = None
preloaded_input_details = None
preloaded_output_details = None

#TODO: implement any useful processes from savepoint version to demo version
#TODO: comment all relevant stuff, even the simple stuff
#loads model using global variables
def load_model():
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details
    print("Loading Object Detector model...")
    try:
        preloaded_interpreter = tflite.Interpreter(
            model_path="./processing/object_detector/best-expanded.tflite")
        preloaded_interpreter.allocate_tensors()
        preloaded_input_details = preloaded_interpreter.get_input_details()
        preloaded_output_details = preloaded_interpreter.get_output_details()
        print("Model loaded")
    except Exception as e:
        print("Preloading OD failed due to: " + str(e))

#Unloads model using global variables
def unload_model():
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details
    print("Unloading Object Detector model...")
    try:
        preloaded_interpreter = None
        preloaded_input_details = None
        preloaded_output_details = None
        print("Model Unloaded")
    except Exception as e:
        print("Unloading OD failed due to: " + str(e))


def process(savepoint):
    """Main function for object detections"""

    #intialises arrays for storing cropped roi's and frames with bb's
    cropped_frames = []
    # annotated_frames = []

    #sets fixed output size, needed for input into keypoint detector
    fixed_box_size = np.asarray([539,561])
    # Loads tensorflowlite model
    interpreter = tflite.Interpreter(
        model_path="./processing/object_detector/best-expanded.tflite")
    # interpreter.resize_tensor_input(
    #     interpreter.get_input_details()[0]['index'],
    #     [1, 640, 640, 3])
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #processes each extracted frame
    for image_name in os.listdir(savepoint):

        #loads original scale image
        true_scale_image = cv2.imread(savepoint+image_name)

        # input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)

        # Creates black canvas, pads to 1280,1280
        fill_color = (0, 0, 0, 255)
        # x, y, c = true_scale_image.shape
        x_target, y_target = 1280, 1280
        new_im = Image.new('RGBA', (x_target, y_target), fill_color)

        #Calculates padding to centre image in sqaure canvas
        original_height, original_width = true_scale_image.shape[:2]
        pos_x = int((x_target - original_width) / 2)
        pos_y = int((y_target - original_height) / 2)

        # pastes original image into centre of the square canvas
        new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (pos_x, pos_y), fill_color)
        expanded_image = np.array(new_im)[..., :3]

        # rescales image to 640, 640 (for model input)
        modified_image = cv2.resize(expanded_image, (640,640))
        #reshapes for model input
        input_data = np.reshape(modified_image, (1, 640, 640, 3))

        max_size = 1920
        ns = 2
        b, h, w, ch = input_data.shape
        scale = max_size / ns / max(h, w)

        #runs the model inference
        interpreter.set_tensor(
            input_details[0]['index'],
            input_data.astype(np.float32))
        interpreter.invoke()

        #collects all output from model into array 'y'
        y = []
        for output in output_details:
            y.append(interpreter.get_tensor(output['index']))
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        # x1, y1, x2, y2, conf, class_index = non_max_suppression(y[0])[0][0]

        #Applies nms
        detections = non_max_suppression(y[0])


        # detections count check, outputs no detections statement
        if len(detections[0]) == 0:
            print(f"No detections found for {image_name} within OD util!")
            continue

        #Wrong. nms output was xyxy
        # x_centre, y_centre, width, height, conf, class_index = detections[0][0]

        x1, y1, x2, y2, conf, class_index = detections[0][0]

        #confidence check, outputs low confidence statement
        if conf < 0.25: #Lowered from 75 to 25
            print(f"Low confidence detection: {conf} within OD util!")
            continue

        ##FOR TRYING TO DEBUG PIPELINE ANNOTATIONS WITHIN A STATIC ENV (BEFORE USING IN REALTIME/DEMO)
        #TODO: BB logic  - Return and attempt to fix after everything else is tied up
        ##print(x1, y1, x2, y2)
        # # x1, y1, x2, y2 = x1*scale, y1*scale, x2*scale, y2*scale
        # #converts to integers for pixel coords
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #
        # # scale back to original size for more accurate BB's
        # scale_factor = 1280 / 640
        # x1_padded = x1 * scale_factor
        # y1_padded = y1 * scale_factor
        # x2_padded = x2 * scale_factor
        # y2_padded = y2 * scale_factor
        #
        # # Removes padding offset to get original coordinates
        # offset_x = (1280 - original_width) / 2
        # offset_y = (1280 - original_height) / 2
        #
        # x1_final = x1_padded - offset_x
        # y1_final = y1_padded - offset_y
        # x2_final = x2_padded - offset_x
        # y2_final = y2_padded - offset_y
        #
        # #sets to image boundaries
        # x1 = max(0, int(x1_final))
        # y1 = max(0, int(y1_final))
        # x2 = min(original_width, int(x2_final))
        # y2 = min(original_height, int(y2_final))

        # annotated_image = true_scale_image.copy()
        # # #TODO: BB's are positionally incorrect. Cannot overcome the issue and think the issue might be within nms
        # # cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #
        # # Adds confidence label
        # confidence_label = f"Internal Confidence: {conf}"
        # cv2.putText(annotated_image, confidence_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #
        # #Add's class label if available
        # if class_index is not None:
        #     class_label = f"Internal Class index: {int(class_index)}"
        #     cv2.putText(annotated_image, class_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #
        # annotated_frames.append(annotated_image)
        # cv2.imwrite(f"./processing/extracted_frames/OD_{image_name}", annotated_image)

        #converts to grayscale before cropping
        gray_true_scale_image = cv2.cvtColor(true_scale_image, cv2.COLOR_BGR2GRAY)


        #creates fixed size crop array
        crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))
        for i in range(crop.shape[0]):
            for j in range(crop.shape[1]):
                ii, jj = y1 + i, y2 + j
                if (ii < gray_true_scale_image.shape[0] and
                        jj < gray_true_scale_image.shape[1]):
                    crop[i][j] = gray_true_scale_image[ii][jj]

        # crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))
        # for i in range(crop.shape[0]):
        #     for j in range(crop.shape[1]):
        #         ii, jj = y1+i, x1+j
        #         if (ii < gray_true_scale_image.shape[0] and
        #             jj < gray_true_scale_image.shape[1]):
        #             crop[i][j] = gray_true_scale_image[ii][jj]

        cropped_frames.append(crop)

    return np.array(cropped_frames)


#Realtime version - slightly modified version of above...with more modifications for better loading
def process_realtime(frame):
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details

    cropped_frames = []

    #MODIFIED to utilise a copy
    true_scale_image = frame.copy()

    fixed_box_size = np.asarray([539, 561])

    #modififed interpreter use utilising preloaded model
    interpreter = preloaded_interpreter
    # interpreter.resize_tensor_input(
    #     interpreter.get_input_details()[0]['index'],
    #     [1, 640, 640, 3])

    # Get input and output tensors.
    input_details = preloaded_input_details
    output_details = preloaded_output_details

    # input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)

    # Creates black canvas, pads to 1280,1280
    fill_color = (0, 0, 0, 255)
    # x, y, c = true_scale_image.shape
    x_target, y_target = 1280, 1280
    new_im = Image.new('RGBA', (x_target, y_target), fill_color)

    # Calculates padding to centre image in sqaure canvas
    original_height, original_width = true_scale_image.shape[:2]
    pos_x = int((x_target - original_width) / 2)
    pos_y = int((y_target - original_height) / 2)

    # pastes original image into centre of the square canvas
    new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (pos_x, pos_y), fill_color)
    expanded_image = np.array(new_im)[..., :3]

    # rescales image to 640, 640 (for model input)
    modified_image = cv2.resize(expanded_image, (640, 640))
    # reshapes for model input
    input_data = np.reshape(modified_image, (1, 640, 640, 3))

    max_size = 1920
    ns = 2
    b, h, w, ch = input_data.shape
    scale = max_size / ns / max(h, w)

    #feeds input details into model and execute it
    interpreter.set_tensor(
        input_details[0]['index'],
        input_data.astype(np.float32))
    interpreter.invoke()

    #extracts detection results
    y = []
    for output in output_details:
        y.append(interpreter.get_tensor(output['index']))
    y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
    y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

    # Applies nms
    # detections = non_max_suppression(y[0])
    x1, y1, x2, y2, conf, class_index = non_max_suppression(y[0])[0][0]

    # # detections count check, outputs no detections statement
    # if len(detections[0]) == 0:
    #     print(f"No detections found for {true_scale_image} within OD util!")
    #     return

    # Wrong. nms output was xyxy
    # x_centre, y_centre, width, height, conf, class_index = detections[0][0]

    # # confidence check, outputs low confidence statement
    # if conf < 0.25:  # Lowered from 75 to 25
    #     print(f"Low confidence detection: {conf} within OD util!")
    #     return

    #Rejects small detections - poor approach to reduce issues regarding small object detections
    #TODO: Remove/Improve this later
    # width = x2 - x1
    # height = y2 - y1
    # if width < 20 or height < 20:
    #     print("Detection too small. False Positive (Potentially)")
    #     return np.array([]), 0, (0, 0, 0, 0)

    # print(x1, y1, x2, y2)
    # x1, y1, x2, y2 = x1*scale, y1*scale, x2*scale, y2*scale
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # #TODO: test out this approach to fixing bbox (essintially undoing preprocessing)
    #get original dimensions


    #scale back to original
    scale_factor = 1280 / 640
    x1_padded = x1 * scale_factor
    y1_padded = y1 * scale_factor
    x2_padded = x2 * scale_factor
    y2_padded = y2 * scale_factor

    #Removes padding offset
    offset_x = (1280 - original_width) / 2
    offset_y = (1280 - original_height) / 2

    x1_final = x1_padded - offset_x
    y1_final = y1_padded - offset_y
    x2_final = x2_padded - offset_x
    y2_final = y2_padded - offset_y

    #sets to image boundaries
    x1 = max(0, int(x1_final))
    y1 = max(0, int(y1_final))
    x2 = min(original_width, int(x2_final))
    y2 = min(original_height, int(y2_final))

    gray_true_scale_image = cv2.cvtColor(true_scale_image, cv2.COLOR_BGR2GRAY)

    #ORIGINAL

    crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            ii, jj = y1 + i, y2 + j
            if (ii < gray_true_scale_image.shape[0] and
                    jj < gray_true_scale_image.shape[1]):
                crop[i][j] = gray_true_scale_image[ii][jj]

    # crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))
    # for i in range(crop.shape[0]):
    #     for j in range(crop.shape[1]):
    #         ii, jj = y1 + i, x1 + j
    #         if (ii < gray_true_scale_image.shape[0] and
    #                 jj < gray_true_scale_image.shape[1]):
    #             crop[i][j] = gray_true_scale_image[ii][jj]

    cropped_frames.append(crop)

    return np.array(cropped_frames), conf, (x1, y1, x2, y2)