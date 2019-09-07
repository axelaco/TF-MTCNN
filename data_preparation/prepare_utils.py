import numpy as np

def IOU(bboxA, bboxB):
    x1_inter = max(bboxA[0], bboxB[0])
    y1_inter = max(bboxA[1], bboxB[1])
    x2_inter = min(bboxA[0] + bboxA[2], bboxB[0] + bboxB[2])
    y2_inter = min(bboxA[1] + bboxA[3], bboxB[1] + bboxB[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    bboxAarea = bboxA[2] * bboxA[3]

    bboxBarea = bboxB[2] * bboxB[3]

    iou_score = inter_area / float(bboxAarea + bboxBarea - inter_area)

    return iou_score

def multiple_IOU(bboxA, bboxes):
    x1_inter = np.maximum(bboxA[0], bboxes[:, 0])
    y1_inter = np.maximum(bboxA[1], bboxes[:, 1])
    x2_inter = np.minimum(bboxA[0] + bboxA[2], bboxes[:, 0] + bboxes[:, 2])
    y2_inter = np.minimum(bboxA[1] + bboxA[3], bboxes[:, 1] + bboxes[:, 3])

    inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)

    bboxAarea = bboxA[2] * bboxA[3]

    bboxesarea = bboxes[:, 2] * bboxes[:, 3]

    iou_score = np.divide(inter_area, bboxAarea + bboxesarea - inter_area)

    return iou_score