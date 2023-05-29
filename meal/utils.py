import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
sns.set_theme()

def apply_gaussian_blur(input_folder, output_folder):
    # define the list of Gaussian blur levels to apply
    blur_levels = [(3, 3), (7, 7), (11, 11), (15, 15)]

    # loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # read in the image
            image_path = os.path.join(input_folder, file_name)
            img = cv2.imread(image_path)

            # apply each level of Gaussian blur and save the resulting images
            for ix, ksize in enumerate(blur_levels):
                img_blur = cv2.GaussianBlur(img, ksize, sigmaX=0)
                output_path = os.path.join(output_folder, f"{file_name[:-4]}.jpg")
                cv2.imwrite(output_path, img_blur)
                

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    return px, py, ap
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        # plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        # plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        # plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def plot_pr_curve(stats):
    names = {
        0: 'L_Vocal Fold',
        1: 'L_Arytenoid cartilage',
        2: 'Benign lesion',
        3: 'Malignant lesion',
        4: 'R_Vocal Fold',
        5: 'R_Arytenoid cartilage'
    }
    color = ['darkgray', 'gray', 'black', 'red', 'blue']
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    for ix, st in enumerate(stats):
        save_dir = "/home/dtpthao/workspace/yolov5"
        px, py, ap = ap_per_class(*st, plot=True, save_dir=save_dir, names=names)
        py = np.stack(py, axis=1)
        if ix == 4:
            label = 'Original (non-blur)'
            ax.plot(px, py.mean(1), linewidth=4, color=color[ix], label=label)
        elif ix == 3:
            label = 'blur {}0%'.format(ix+1)
            ax.plot(px, py.mean(1), linewidth=4, color=color[ix], label=label)
        else:
            label = 'blur {}0%'.format(ix+1)
            ax.plot(px, py.mean(1), linewidth=2, color=color[ix], label=label)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(os.path.join(save_dir, "PR_curve.png"), dpi=300)
    plt.close(fig)


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)
        

def get_binary_mask(shape, boxes, fill_value=1):
    """
    Generates a binary mask tensor of size `shape` with values set to `fill_value` inside
    the bounding boxes specified by `boxes`, and 0 outside the boxes.
    
    Args:
    - shape (tuple[int, int]): The shape of the output binary mask tensor.
    - boxes (list[list[int]]): A list of bounding box coordinates in the format [x, y, w, h].
    - fill_value (float): The value to fill inside the bounding boxes. Default is 1.
    
    Returns:
    - binary_mask (torch.Tensor): A binary mask tensor of size `shape`.
    """
    binary_mask = torch.zeros(shape)
    for box in boxes:
        x1, y1, x2, y2 = box
        y1 = int(y1 / 360.0 * 480.0)
        y2 = int(y2 / 360.0 * 480.0)
        binary_mask[y1:y2, x1:x2] = fill_value
    return binary_mask