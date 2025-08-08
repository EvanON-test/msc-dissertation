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
    annotated_frames = []

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
        x, y, c = true_scale_image.shape
        x_target, y_target = 1280, 1280
        new_im = Image.new('RGBA', (x_target, y_target), fill_color)
        # # pos_x = (int((x_target - x) / 2))
        # pos_y = (int((y_target - x) / 2))
        # new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (0, pos_y))
        # expanded_image = np.array(new_im)[...,:3]

        #Calculates padding to centre image in sqaure canvas
        original_height, original_width = true_scale_image.shape[:2]
        pos_x = int((1280 - original_width) / 2)
        pos_y = int((1280 - original_height) / 2)

        # pastes original image into centre of the square canvas
        new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (pos_x, pos_y))
        expanded_image = np.array(new_im)[..., :3]

        print(f"DEBUGGING FOR: {image_name}")
        print(f"Original HxW: {original_height} x {original_width}")
        print(f"Padding Pos_x, pos_y: {pos_x}, {pos_y}")
        print(f"Expanded shape: {expanded_image.shape}")

        # # GPU - rescale to 640, 640
        # try:
        #     modified_image = rescale_image_gpu(expanded_image)
        #     print("GPU used")
        # except Exception as e :

        # rescales image to 640, 640 (for model input)
        modified_image = cv2.resize(expanded_image, (640,640))
        #reshapes for model input
        input_data = np.reshape(modified_image, (1, 640, 640, 3))

        max_size = 1920
        ns = 2
        b, h, w, ch = input_data.shape
        scale = max_size / ns / max(h, w)

        print(f"Resized: {modified_image.shape}, Input shape: {input_data.shape} (h={h}, w={w})")

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

        print(f"\ny[0] shape: {y[0].shape}")



        # x1, y1, x2, y2, conf, class_index = non_max_suppression(y[0])[0][0]
        #Applies nms
        detections = non_max_suppression(y[0])

        print(f"nms batches {len(detections)}, dets in batch {len(detections[0]) if len(detections)>0 else 0}")
        if len(detections[0]) > 0:
            det = detections[0][0]
            print(f"DETECTIONS RAW: {det[:6].tolist()}")


        # detections count check, outputs no detections statement
        if len(detections[0]) == 0:
            print(f"No detections found for {image_name} within OD util!")
            continue

        #Wrong. nms output was xyxy
        # x_centre, y_centre, width, height, conf, class_index = detections[0][0]

        x1, y1, x2, y2, conf, class_index = detections[0][0]


        print(f"640 space xyxy: ({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f}), size {x2-x1:.2f}x{y2-y1:.2f}, conf {conf:.2f}, cls {int(class_index) if class_index is not None else 'NA'}")

        # print(f"\nDEBUGGING OUTPUTS FOR: {image_name}")
        # print(f"BBOX: ({x1}, {y1}), ({x2}, {y2})")
        # print(f"Original image size {original_width}x{original_height}")
        # print(f'Size: {width}, height: {height}')
        # print(f"Confidence: {conf}")


        #confidence check, outputs low confidence statement
        if conf < 0.25: #Lowered from 75 to 25
            print(f"Low confidence detection: {conf} within OD util!")
            continue
        #As output was as origianlly thought xyxy it doesnt need this
        # x1 = x_centre - width / 2
        # y1 = y_centre - height / 2
        # x2 = x_centre + width / 2
        # y2 = y_centre + height / 2


        # # print(x1, y1, x2, y2)
        # # x1, y1, x2, y2 = x1*scale, y1*scale, x2*scale, y2*scale
        # #converts to integers for pixel coords
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #
        # scale back to original size
        scale_factor = 1280 / 640
        x1_padded = x1 * scale_factor
        y1_padded = y1 * scale_factor
        x2_padded = x2 * scale_factor
        y2_padded = y2 * scale_factor

        # Removes padding offset to get original coordinates
        offset_x = (1280 - original_width) / 2
        offset_y = (1280 - original_height) / 2

        print(f"scale factor: {scale_factor}, offsets: {offset_x:.2f}, {offset_y:.2f}")
        print(f"padded(1280) xyxy: ({x1_padded:.2f}, {y1_padded:.2f}) - ({x2_padded:.2f}, {y2_padded:.2f})")
        print(f"offset_x: {offset_x:.2f}, offset_y: {offset_y:.2f}")

        x1_final = x1_padded - offset_x
        y1_final = y1_padded - offset_y
        x2_final = x2_padded - offset_x
        y2_final = y2_padded - offset_y

        print(f"unpadded to original xyxy: ({x1_final:.2f}, {y1_final:.2f}) - ({x2_final:.2f}, {y2_final:.2f})")

        #sets to image boundaries
        x1 = max(0, int(x1_final))
        y1 = max(0, int(y1_final))
        x2 = min(original_width, int(x2_final))
        y2 = min(original_height, int(y2_final))

        actual_box_width = x2 - x1
        actual_box_height = y2 - y1

        print(f"Final int bbox: ({x1}, {y1}) - ({x2}, {y2}) on {original_width} x {original_height}, size {x2-x1}x{y2-y1}")
        if x1>=x2 or y1 >= y2:
            print("WARNING: invalid/zero area bbox after setting to image boundaries")
        # print(f"BBox: ({x1, y1}, {x2, y2})")
        # print(f"Size after nms: {actual_box_width}x{actual_box_height}")
        # print(f"Original image size {original_width}x{original_height}")

        annotated_image = true_scale_image.copy()
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Adds confidence label
        confidence_label = f"Internal Confidence: {conf}"
        cv2.putText(annotated_image, confidence_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #Add's class label if available
        if class_index is not None:
            class_label = f"Internal Class index: {int(class_index)}"
            cv2.putText(annotated_image, class_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        annotated_frames.append(annotated_image)
        cv2.imwrite(f"./processing/extracted_frames/OD_{image_name}", annotated_image)



        #TODO: CONTINUE FROM HERE TOMORROW ....IT'S GETTING CLOSE BUT NOT QUITE YET




        # # #TODO: this creates a better sized box - but not in correct location
        # fb0 = fixed_box_size[0]//2
        # fb1 = fixed_box_size[1]//2
        #
        # start = (x1, y1)
        # # end = (y1+fb0, y2+fb1)
        # end = (x2+fb1, y2+fb0)
        # cv2.rectangle(modified_image, start, end, (0,255,0), 3)
        # cv2.imwrite(f"./processing/extracted_frames/OD_{image_name}", modified_image)


        #TODO: FIX BOUNDING HERE FIRST BEFORE MOVING BACK TO REALTIME - CHANGES HAVE HELPED BUT STILL WRONG
        #draws green bounding box on compy of original frame
        # annotated_image = true_scale_image.copy()
        # cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


        #Adds confidence label
        # confidence_label = f"Internal Confidence: {conf}"
        # cv2.putText(annotated_image, confidence_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #
        # #Add's class label if available
        # if hasattr(class_index, '__len__') or class_index is not None:
        #     class_label = f"Internal Class index: {int(class_index)}"
        #     cv2.putText(annotated_image, class_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #
        # #appends annotated image to annotated frames and also outputs it to the defined directory
        # annotated_frames.append(annotated_image)
        # cv2.imwrite(f"./processing/extracted_frames/OD_{image_name}", annotated_image)

        # fb0 = fixed_box_size[0]//2
        # fb1 = fixed_box_size[1]//2

        #TODO: TRY THIS IF IT FAILS
        #start = (y1, y2)
        #end = (y1+fb0, y2+fb1)
        #plot = cv2.rectangle(modified_image, start, end, (0,255,0), 3)
    
        #converts to grayscale before cropping
        gray_true_scale_image = cv2.cvtColor(true_scale_image, cv2.COLOR_BGR2GRAY)

        #creates fixed size crop array
        crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))

        # for i in range(crop.shape[0]):
        #     for j in range(crop.shape[1]):
        #         ii, jj = y1+i, y2+j
        #         if (ii < gray_true_scale_image.shape[0] and
        #             jj < gray_true_scale_image.shape[1]):
        #             crop[i][j] = gray_true_scale_image[ii][jj]

        #TODO: TEST
        # crop = np.zeros((fixed_box_size[0], fixed_box_size[1]))
        for i in range(crop.shape[0]):
            for j in range(crop.shape[1]):
                ii, jj = y1+i, x1+j
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
    #TODO: test these modifications
    #---
    #pos_y = (int((y_target - x) / 2))
    #
    # #pastes into centre of the square canvas
    # new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (0, pos_y))
    #---
    original_height, original_width = true_scale_image.shape[:2]
    pos_x = int((1280 - original_width) / 2)
    pos_y = int((1280 - original_height) / 2)

    #pastes into centre of the square canvas
    new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (pos_x, pos_y))
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
    #TODO: Remove/Improve this later
    width = x2 - x1
    height = y2 - y1
    if width < 20 or height < 20:
        print("Detection too small. False Positive (Potentially)")
        return np.array([]), 0, (0, 0, 0, 0)

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

    x1 = max(0, int(x1_final))
    y1 = max(0, int(y1_final))
    x2 = min(original_width, int(x2_final))
    y2 = min(original_height, int(y2_final))

    # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


    # fb0 = fixed_box_size[0] // 2
    # fb1 = fixed_box_size[1] // 2

    # start = (y1, y2)
    # end = (y1 + fb0, y2 + fb1)
    # plot = cv2.rectangle(modified_image, start, end, (0, 255, 0), 3)

    gray_true_scale_image = cv2.cvtColor(true_scale_image, cv2.COLOR_BGR2GRAY)

    #TODO: test the for loop change and return change
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

    # cv2.imshow("crop", crop.astype(np.uint8))
    # cv2.imshow("plot", plot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cast coords to int, draw box on image
    # showw("img", plot)

    #tried fiddling with return
    #ORIGINAL:   np.array(cropped_frames)
    #crop.astype(np.uint8)

    return np.array(cropped_frames), conf, (x1, y1, x2, y2)