from __future__ import absolute_import, division

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def best_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers. In this case, predicted consists
    of multiple predictions and the best one will be chosen.
    """

    target = target.unsqueeze(-1)
    diff = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 2), dim=1)
    best, _ = torch.min(diff, dim=1)

    return torch.mean(best)


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def best_p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers. In this case, predicted consists
    of multiple predictions and the best one will be chosen.
    """

    for i in range(predicted.shape[-1]):
        pred = predicted[:, :, :, i]
        err = p_mpjpe(pred, target, reduce=None)
        try:
            best[err < best] = err[err < best]
        except:
            best = err

    return np.mean(best)
        

def p_mpjpe(predicted, target, reduce='mean'):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    if reduce == 'mean':
        return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))
    else:
        return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)

def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))


class MixedCycleLoss(nn.Module):

    def __init__(self, reduction: str = 'none') -> None:
        super(MixedCycleLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input_2d, input_3d, target_2d, target_3d, w_cycle = 1, w_3d = 1):
        # pdb.set_trace()
        loss_cycle = F.mse_loss(input_2d, target_2d, reduction=self.reduction)
        loss_3d = F.mse_loss(input_3d, target_3d, reduction=self.reduction)
        mixed_loss = w_cycle * loss_cycle + w_3d * loss_3d
        return mixed_loss, loss_cycle, loss_3d
