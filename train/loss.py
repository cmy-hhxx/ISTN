# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class AutoencoderLoss(nn.Module):
    def __init__(self):
        super(AutoencoderLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, recon_x, x):
        original_shape = (-1, 1, 256, 256)  # 恢复到四维形状

        # 重塑张量以适应 SSIM 输入需求
        recon_x_reshaped = recon_x.view(original_shape)
        x_reshaped = x.view(original_shape)

        mse = self.mse_loss(recon_x_reshaped, x_reshaped)
        ssim_out = 1 - self.ssim_loss(recon_x_reshaped, x_reshaped)
        loss = mse + 0.5 * ssim_out
        return loss


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

    def forward(self, seg_output, target_masks):
        """
        seg_output: 分割模型输出，shape [batch_size, num_classes, height, width]
        target_masks: 目标掩模，shape [batch_size, height, width]，包含不同结构的标签
        """
        # 分类损失（交叉熵损失）
        # 将 target_masks 转化为 long 类型，因为交叉熵损失需要整数标签
        target_masks = target_masks.long()
        ce_loss = F.cross_entropy(seg_output, target_masks)

        # 平滑性约束，鼓励输出的空间连续性，减少孤立的像素点
        dx, dy = seg_output[:, :, :-1, :] - seg_output[:, :, 1:, :], seg_output[:, :, :, :-1] - seg_output[:, :, :, 1:]
        dx_loss = dx.abs().mean()
        dy_loss = dy.abs().mean()
        smoothness_loss = dx_loss + dy_loss

        # 总损失
        total_loss = ce_loss + 0.1 * smoothness_loss

        return total_loss

# class SegmentationLoss(nn.Module):
#     def __init__(self):
#         super(SegmentationLoss, self).__init__()
#
#     def forward(self, seg_output, features_A, features_B):
#         # 假设 seg_output 的通道数为 2，对应结构 A 和 B
#         seg_A = seg_output[:, 0, :, :]
#         seg_B = seg_output[:, 1, :, :]
#
#         # 计算分割掩码与特征图之间的相关性
#         loss_A = -torch.mean(F.cosine_similarity(seg_A.unsqueeze(1), features_A, dim=1))
#         loss_B = -torch.mean(F.cosine_similarity(seg_B.unsqueeze(1), features_B, dim=1))
#
#         # 添加平滑性约束
#         smoothness_loss = torch.mean(torch.abs(seg_A[:, :, :-1] - seg_A[:, :, 1:])) + \
#                           torch.mean(torch.abs(seg_A[:, :-1, :] - seg_A[:, 1:, :])) + \
#                           torch.mean(torch.abs(seg_B[:, :, :-1] - seg_B[:, :, 1:])) + \
#                           torch.mean(torch.abs(seg_B[:, :-1, :] - seg_B[:, 1:, :]))
#
#         # 总损失
#         total_loss = loss_A + loss_B + 0.1 * smoothness_loss
#
#         return total_loss
