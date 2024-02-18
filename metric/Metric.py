
import numpy as np
from sklearn.metrics import precision_recall_curve

######## metric ###########
def metric(premask, groundtruth):
    groundtruth = groundtruth.flatten()
    premask = premask.flatten() 
    groundtruth = groundtruth.cpu().numpy()
    premask = premask.cpu().numpy()

    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        print('cross+union=0!!!!')
        iou = 1
    return f1, iou

######## metric ###########
def metric_numpy(premask, groundtruth):
    groundtruth = groundtruth.flatten()
    premask = premask.flatten()
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    #fnr = false_neg/(true_pos+false_neg + 1e-6)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        #print('cross+union=0!!!!')
        iou = 1
    return f1, iou




######## metric ###########
def metric_numpy_test(premask, groundtruth):
    groundtruth = groundtruth.flatten()
    premask = premask.flatten()

    # # 计算精确度和召回率
    # precision, recall, _ = precision_recall_curve(groundtruth, premask)
    # print(precision.shape)
    # print(recall.shape)
    # exit()
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    fnr = false_neg/(true_pos+false_neg + 1e-6)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        #print('cross+union=0!!!!')
        iou = 1
    return f1, iou,fnr


# auc
def metric_auc(premask, groundtruth):
    groundtruth = groundtruth.flatten()
    premask = premask.flatten()
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        print('cross+union=0!!!!')
        iou = 1
    return f1, iou