import torch
import torch.nn as nn


def get_dice_loss(gt_score, pred_score):
	inter = torch.sum(gt_score * pred_score)
	union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
	return 1. - (2 * inter / union)


class Loss(nn.Module):
	def __init__(self, weight_angle=10):
		super(Loss, self).__init__()
		self.weight_angle = weight_angle

	def forward(self, gt_score, pred_score):
		if torch.sum(gt_score) < 1:
			return torch.sum(pred_score) * 0
		
		classify_loss = get_dice_loss(gt_score, pred_score)

		return classify_loss
