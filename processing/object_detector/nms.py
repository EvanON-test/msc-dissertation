import numpy as np
import time

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def calculate_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    iou = intersection_area / (box_area + boxes_area - intersection_area)

    return iou

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=1,
    nm=0,  # number of masks
):
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    mi = 5 + nc
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].astype(int) + 5] = 1.0
            x = np.concatenate((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].astype(float), mask[i]), 1)
        else:
            conf = x[:, 5:mi].max(1)
            j = x[:, 5:mi].argmax(1)
            j = j.reshape(-1, 1)
            result = np.concatenate((box, conf.reshape(-1, 1), j.astype(float), mask), axis=1)
            conf_mask = conf > conf_thres
            x = result[conf_mask]

        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        n = x.shape[0]

        if not n:
            continue

        sorted_indices = np.argsort(x[:, 4])[::-1]
        sorted_x = x[sorted_indices]
        x = sorted_x[:max_nms]

        selected_indices = []

        for i in range(len(x)):
            iou = calculate_iou(x[i], x)
            iou[i] = 0  # Ignore self-iou
            mask = iou <= iou_thres
            x[i, 4] *= np.prod(mask)

            if x[i, 4] > 0:
                selected_indices.append(i)

        selected_indices = selected_indices[:max_det]

        if merge and (1 < n < 3E3):
            iou = calculate_iou(x[selected_indices], x)
            weights = iou * x[selected_indices, 4][:, None]
            x[selected_indices, :4] = np.dot(weights, x[:, :4]) / weights.sum(1, keepdims=True)
            
            if redundant:
                iou = calculate_iou(x[selected_indices], x[selected_indices])
                redundant_indices = iou.sum(1) > 1
                selected_indices = [i for i, r in enumerate(redundant_indices) if not r]

        output[xi] = x[selected_indices]

    return output

# Example usage:
# Replace 'prediction', 'conf_thres', and 'iou_thres' with your actual data and thresholds
# detections = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45)
