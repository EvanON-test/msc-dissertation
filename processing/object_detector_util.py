from processing.object_detector.nms import non_max_suppression
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import time
import cv2
import sys
import os


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