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

# #TODO: REATTEMPT GPU APPROACH AGAIN
BATCH_SIZE = 1
# GPU_BATCH_SIZE = 8

# CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
# if CUDA_AVAILABLE:
#     try:
#         print("CUDA initialised successfully")
#     except Exception as e:
#         print("CUDA not initialised due to:" + str(e))

# # TODO: This is clearly wrong (NOT proper batch). Paused 14/07
# def rescale_image_gpu(image):
#     try:
#         if image is None or image.size == 0:
#             raise Exception("Image is empty")
#         # gpu_grey = cv2.cuda_GpuMat()
#         # cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY, dst=gpu_grey)
#         grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         gpu_image = cv2.cuda_GpuMat()
#         gpu_resized = cv2.cuda_GpuMat()
#
#         gpu_image.upload(grey_image)
#         cv2.cuda.resize(gpu_image, (LOW_RES_WIDTH, LOW_RES_HEIGHT), dst=gpu_resized)
#         result = gpu_resized.download()
#         return result
#     except Exception as e:
#         print("GPU processing failed due to: " + str(e))

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
        print("\nERROR\n")
        sys.exit()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    preds = np.zeros((total_frames))

    #processes the video in batches (1 currently)
    for i in range(0, total_frames, BATCH_SIZE):
        # ignore channels as images will be grayscale
        batch = np.zeros((BATCH_SIZE, LOW_RES_HEIGHT, LOW_RES_WIDTH))
        for b in range(0, BATCH_SIZE):
            if success:
                # TODO: Return to this approach and try to implement it alongside the TensorRT approach?
                # GPU - grayscale and rescale
                # try:
                #     batch[b] = rescale_image_gpu(image)
                #     success, image = video.read()
                #     # print("GPU utilisation CONFIRMED!")
                # except Exception as e :
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


# def get_centroid(s):
#     return (np.max(s)+np.min(s))/2
#
#
# def subplot_curves(ax, preds, *curves):
#     ax.plot(np.arange(0,preds.shape[0]), preds)
#     for curve in curves:
#         ax.plot(np.arange(0,preds.shape[0]), curve)


# def plot_curves(preds, s1, s2, r1, r2):
#     axlabelsize = 16
#     fig, ax = plt.subplots(3)
#     ax[0].set_title("Raw Classifier Output", fontsize=axlabelsize)
#     ax[0].plot(np.arange(0,preds.shape[0]), preds, 'k-')
#
#     stem_width = 2
#
#     # _, stemlines, _ = ax[0].stem(
#     #     [10, 225, 467], [1.1, 1.1, 1.1],
#     #     linefmt ='k-', markerfmt='Dk', basefmt='k')
#     # plt.setp(stemlines, 'linewidth', stem_width)
#     # _, stemlines, _ = ax[0].stem(
#     #     [42, 288, 560], [1.1, 1.1, 1.1],
#     #     linefmt ='k-', markerfmt='ks', basefmt='k')
#     # plt.setp(stemlines, 'linewidth', stem_width)
#
#     ax[1].set_title("First Smoothing and Step Function",
#         fontsize=axlabelsize)
#     ax[1].plot(np.arange(0,preds.shape[0]), preds, 'k-')
#     ax[1].plot(np.arange(0,preds.shape[0]), s1, 'k:')
#     ax[1].plot(np.arange(0,preds.shape[0]), r1, 'k--')
#
#     # _, stemlines, _ = ax[1].stem(
#     #     [10, 225, 467], [1.1, 1.1, 1.1],
#     #     linefmt ='k-', markerfmt='Dk', basefmt='k')
#     # plt.setp(stemlines, 'linewidth', stem_width)
#     # _, stemlines, _ = ax[1].stem(
#     #     [42, 288, 560], [1.1, 1.1, 1.1],
#     #     linefmt ='k-', markerfmt='ks', basefmt='k')
#     # plt.setp(stemlines, 'linewidth', stem_width)
#
#     ax[2].set_title("Second Smoothing and Step Function",
#         fontsize=axlabelsize)
#     ax[2].plot(np.arange(0,preds.shape[0]), preds, 'k-',
#         label='classifier output')
#     ax[2].plot(np.arange(0,preds.shape[0]), s2, 'k:',
#         label='smoothed value')
#     ax[2].plot(np.arange(0,preds.shape[0]), r2, 'k--',
#         label='rounded value')
#
#     # _, stemlines, _ = ax[2].stem(
#     #     [10, 225, 467], [1.1, 1.1, 1.1],
#     #     linefmt ='k-', markerfmt='Dk', basefmt='k',
#     #     label='animal enters scene')
#     # plt.setp(stemlines, 'linewidth', stem_width)
#     # _, stemlines, _ = ax[2].stem(
#     #     [42, 288, 560], [1.1, 1.1, 1.1],
#     #     linefmt ='k-', markerfmt='ks', basefmt='k',
#     #     label='animal leaves scene')
#     # plt.setp(stemlines, 'linewidth', stem_width)
#
#     fig.legend(loc='center right', fontsize=14)
#     fig.text(0.5, 0.07, 'Frame', ha='center', fontsize=axlabelsize)
#     fig.text(0.08, 0.5, 'Prediction', va='center', rotation='vertical',
#         fontsize=axlabelsize)
#     plt.show()





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

    # for i in range(len(r2)):
    #     cv2.imshow("ref", frames[i].astype(np.uint8))
    #     key = cv2.waitKeyEx(0)
    #     if key == 32: # space
    #         print(i)
    # cv2.destroyAllWindows()

    # show predictions in various stages of smoothing
    # plot_curves(preds, s1, s2, r1, r2)
    # sys.exit()

    # unload model after use
    del model

    # set back to start of video
    # video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # success, image = video.read()
    # total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # for i in range(0, total_frames):
    #     print(r2[i])
    #     cv2.imshow("image", image)
    #     cv2.waitKey(0)
    #     success, image = video.read()

    # sys.exit()

    return r2

