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

#TODO: implement any useful changes from realtime version into process(avepoint version)
#TODO: comment your stuff, even the simple stuff
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

    cropped_frames = []

    fixed_box_size = np.asarray([539,561])
    
    interpreter = tflite.Interpreter(
        model_path="./processing/object_detector/best-expanded.tflite")
    # interpreter.resize_tensor_input(
    #     interpreter.get_input_details()[0]['index'], 
    #     [1, 640, 640, 3])
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for image_name in os.listdir(savepoint):

        true_scale_image = cv2.imread(savepoint+image_name)

        # input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)

        # pad to 1280,1280
        fill_color = (0, 0, 0, 255)
        x, y, c = true_scale_image.shape
        x_target, y_target = 1280, 1280
        new_im = Image.new('RGBA', (x_target, y_target), fill_color)
        # pos_x = (int((x_target - x) / 2))
        pos_y = (int((y_target - x) / 2))
        new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (0, pos_y))
        expanded_image = np.array(new_im)[...,:3]


        # # GPU - rescale to 640, 640
        # try:
        #     modified_image = rescale_image_gpu(expanded_image)
        #     print("GPU used")
        # except Exception as e :
        # rescale to 640, 640
        modified_image = cv2.resize(expanded_image, (640,640))

        input_data = np.reshape(modified_image, (1, 640, 640, 3))

        max_size = 1920
        ns = 2
        b, h, w, ch = input_data.shape
        scale = max_size / ns / max(h, w)

        interpreter.set_tensor(
            input_details[0]['index'], 
            input_data.astype(np.float32))
        interpreter.invoke()

        y = []
        for output in output_details:
            y.append(interpreter.get_tensor(output['index']))
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        x1, y1, x2, y2, conf, class_index = non_max_suppression(y[0])[0][0]

        # print(x1, y1, x2, y2)
        # x1, y1, x2, y2 = x1*scale, y1*scale, x2*scale, y2*scale
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        fb0 = fixed_box_size[0]//2
        fb1 = fixed_box_size[1]//2

        start = (y1, y2)
        end = (y1+fb0, y2+fb1)
        plot = cv2.rectangle(modified_image, start, end, (0,255,0), 3)
    
        gray_true_scale_image = cv2.cvtColor(true_scale_image, cv2.COLOR_BGR2GRAY)

        crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))
        for i in range(crop.shape[0]):
            for j in range(crop.shape[1]):
                ii, jj = y1+i, y2+j
                if (ii < gray_true_scale_image.shape[0] and 
                    jj < gray_true_scale_image.shape[1]):
                    crop[i][j] = gray_true_scale_image[ii][jj]

        cropped_frames.append(crop)


        # cv2.imshow("crop", crop.astype(np.uint8))
        # cv2.imshow("plot", plot)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # cast coords to int, draw box on image
        # showw("img", plot)

    return np.array(cropped_frames)


#Realtime version - slightly modified version of above...with more modifications for better loading
def process_realtime(frame):
    global preloaded_interpreter
    global preloaded_input_details
    global preloaded_output_details

    cropped_frames = []

    #MODIFIED to utilise the frame and not the saved image
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

    # pad to 1280,1280 to make a square
    fill_color = (0, 0, 0, 255)
    x, y, c = true_scale_image.shape
    x_target, y_target = 1280, 1280

    #creates a sqaure canvas
    new_im = Image.new('RGBA', (x_target, y_target), fill_color)

    #calculatesvertical centering offset
    # pos_x = (int((x_target - x) / 2))
    pos_y = (int((y_target - x) / 2))

    #pastes into centre of the square canvas
    new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (0, pos_y))
    expanded_image = np.array(new_im)[..., :3]


    modified_image = cv2.resize(expanded_image, (640, 640))

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

    #Apply NMS, seperate file, to filter overlapping detections
    x1, y1, x2, y2, conf, class_index = non_max_suppression(y[0])[0][0]

    #Rejects small detections - poor approach to reduce issues regarding small object detections
    #TODO: Improve this later
    width = x2 - x1
    height = y2 - y1
    if width < 20 or height < 20:
        print("Detection too small. False Positive (Potentially)")
        return np.array([]), 0, (0, 0, 0, 0)

    # print(x1, y1, x2, y2)
    # x1, y1, x2, y2 = x1*scale, y1*scale, x2*scale, y2*scale
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    #TODO: test out this approach to fixing bbox (essintially undoing preprocessing)
    #get original dimensions
    original_height, original_width = true_scale_image.shape[:2]

    #scale back to original
    scale_factor = 1280 / 640
    x1, y1, x2, y2 = x1*scale_factor, y1*scale_factor, x2*scale_factor, y2*scale_factor

    #Removes padding offset
    pos_y = ((y_target - x) / 2)
    y1 = y1 - pos_y
    y2 = y2 - pos_y

    #Scale back to original width
    x1 = x1 * (original_width / 1280)
    x2 = x2 * (original_width / 1280)

    x1 = max(0, min(int(x1), original_width))
    y1 = max(0, min(int(y1), original_height))
    x2 = max(0, min(int(x2), original_width))
    y2 = max(0, min(int(y2), original_height))


    fb0 = fixed_box_size[0] // 2
    fb1 = fixed_box_size[1] // 2

    start = (y1, y2)
    end = (y1 + fb0, y2 + fb1)
    plot = cv2.rectangle(modified_image, start, end, (0, 255, 0), 3)

    gray_true_scale_image = cv2.cvtColor(true_scale_image, cv2.COLOR_BGR2GRAY)

    crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            ii, jj = y1 + i, y2 + j
            if (ii < gray_true_scale_image.shape[0] and
                    jj < gray_true_scale_image.shape[1]):
                crop[i][j] = gray_true_scale_image[ii][jj]

    cropped_frames.append(crop)

    # cv2.imshow("crop", crop.astype(np.uint8))
    # cv2.imshow("plot", plot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cast coords to int, draw box on image
    # showw("img", plot)

    return np.array(cropped_frames), conf, (x1, y1, x2, y2)