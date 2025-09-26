'''
@Project ：MG-Net 
@File    ：tools.py
@IDE     ：PyCharm 
@Author  ：王一梵
@Date    ：2025/9/26 11:13 
'''

import numpy as np
import torch
import cv2
from torch import nn
import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lr_adj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lr_adj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def create_rgb_image(hyperspectral_image, bands):
    """
    从高光谱图像的指定波段创建RGB图像。

    :param hyperspectral_image: 高光谱图像 (波段, 高, 宽)
    :param bands: 用于创建RGB的波段索引 (list)
    :return: RGB图像 (高, 宽, 3)
    """
    return np.transpose(hyperspectral_image[bands, :, :], (1, 2, 0))


def compute_optical_flow(rgb_image1, rgb_image2):
    """
    计算两个RGB图像之间的光流。

    :param rgb_image1: 第一个RGB图像
    :param rgb_image2: 第二个RGB图像
    :return: 光流场
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(rgb_image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(rgb_image2, cv2.COLOR_RGB2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def compute_overall_flow(hyperspectral_data):
    # 创建RGB图像
    rgb_images = []
    for i in range(hyperspectral_data.shape[0] - 2):
        rgb_image = create_rgb_image(hyperspectral_data, [i, i + 1, i + 2])
        rgb_images.append(rgb_image)

    # 计算相邻RGB图像之间的光流
    optical_flows = []
    for i in range(len(rgb_images) - 1):
        flow = compute_optical_flow(rgb_images[i], rgb_images[i + 1])
        optical_flows.append(flow)

    optical_flows = np.array(optical_flows)

    optical_flows_magnitude = np.linalg.norm(optical_flows, axis=3)

    return optical_flows_magnitude

def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)
    return abundance_loss

def SADloss_weigh(output, target, matrix):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    matrix = torch.reshape(matrix, (-1, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0)) * (1 / matrix)
    abundance_loss = torch.mean(abundance_loss)
    return abundance_loss


MSE = nn.MSELoss(reduction='mean')


def rmse_loss(input_tensor, target_tensor, weight_matrix):
    return torch.sqrt(torch.mean((input_tensor - target_tensor) ** 2 * (1 / weight_matrix)))


def MIC(E, lamb):
    """
    计算端元矩阵 E 的 Mutual Incoherence Constraint (MIC) 正则项
    :param E: 端元矩阵 (m x n)，其中 m 是波段数，n 是端元数
    :return: MIC 正则项的值
    """
    # 初始化 MIC 值
    mic_value = 0.0
    num_endmembers = E.shape[1]  # 列数，即端元个数

    # 遍历所有的端元对 (i, j)
    for i in range(num_endmembers):
        for j in range(i + 1, num_endmembers):  # 避免计算 i == j 的情况
            # 计算 E_i 和 E_j 的内积
            dot_product = np.dot(E[:, i], E[:, j])

            # 计算 E_i 和 E_j 的 L2 范式
            norm_i = np.linalg.norm(E[:, i], 2)
            norm_j = np.linalg.norm(E[:, j], 2)

            # 累加 MIC 项
            mic_value += np.abs(dot_product) / (norm_i * norm_j)

    return lamb * mic_value


def fisher_loss(output, labels, num_classes):
    # 类内散布矩阵
    Sw = 0
    Sb = 0
    total_mean = torch.mean(output, dim=0)

    for i in range(num_classes):
        class_samples = output[labels == i]
        class_mean = torch.mean(class_samples, dim=0)

        Sw += torch.sum((class_samples - class_mean).pow(2))
        Sb += class_samples.size(0) * (class_mean - total_mean).pow(2).sum()

    loss = Sw / Sb
    return loss


def min_max_normalize_rows(data):
    """对数据的每一行进行最小-最大归一化，将数据缩放到[0, 1]范围"""
    min_vals = np.amin(data, axis=1, keepdims=True)
    max_vals = np.amax(data, axis=1, keepdims=True)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


def spectral_angle_distance_matrix(outputs, labels):
    """计算输出和标签之间的光谱角距离矩阵"""
    n_outputs = outputs.shape[0]
    n_labels = labels.shape[0]
    distance_matrix = np.zeros((n_outputs, n_labels))

    for i in range(n_outputs):
        for j in range(n_labels):
            dot_product = np.dot(outputs[i], labels[j])
            norm_outputs = np.linalg.norm(outputs[i])
            norm_labels = np.linalg.norm(labels[j])
            cos_theta = dot_product / (norm_outputs * norm_labels + 1e-9)
            distance_matrix[i, j] = np.arccos(np.clip(cos_theta, -1, 1))

    return distance_matrix


def SAD(E_pre, E_true):
    # assert spectral1.shape != spectral2.shape, "两者尺寸不一"
    eps = 1e-8
    SADs = []
    for i in range(E_true.shape[1]):
        core = np.dot(E_true[:, i].T, E_pre[:, i])
        norm_spectral1 = np.linalg.norm(E_true[:, i])
        norm_spectral2 = np.linalg.norm(E_pre[:, i])
        deno = np.dot(norm_spectral1, norm_spectral2.T)
        SAD_idx = np.arccos(((core + eps) / (deno + eps)).clip(-1, 1))
        print(f'第{i + 1}个端元的SAD：{SAD_idx}')
        SADs.append(SAD_idx)
    print(f'平均SAD：{np.mean(SADs)}')

def lhalf_pixelwise_loss(A, eps=1e-8, reduction='sum'):
    """
    逐像素 L_{1/2}: 对每像素的通道能量先求 L2，再开 sqrt。
    A: [C,W,H] or [B,C,W,H]
    """
    is_batched = (A.dim() == 4)
    if not is_batched:
        A = A.unsqueeze(0)
    B, C, W, H = A.shape
    # [B, W, H] 每个像素的通道能量平方和
    perpix = torch.sqrt((A**2).sum(dim=1) + eps)   # L2，已含 eps
    # 再对每像素的 L2 开 sqrt 得到 (||a_p||_2)^{1/2}
    val = torch.sqrt(perpix + eps)                 # [B,W,H]
    if reduction == 'sum':
        return val.sum()
    elif reduction == 'mean':
        return val.mean()
    elif reduction == 'batch_mean':
        return val.view(B, -1).sum(dim=1).mean()
    else:
        raise ValueError("reduction must be 'sum' | 'mean' | 'batch_mean'")

def l21_penalty(A, eps=1e-12, reduction='sum'):
    """
    L2,1 penalty on abundance A.
    A: tensor with shape [C, W, H] or [B, C, W, H]
    Returns scalar tensor.
    reduction: 'sum' or 'mean' over channels.
    """
    is_batched = (A.dim() == 4)
    if not is_batched:
        A = A.unsqueeze(0)  # [1, C, W, H]
    B, C, W, H = A.shape
    # flatten spatial dims
    A_flat = A.view(B, C, -1)            # [B, C, W*H]
    # L2 norm per channel per batch
    norms = torch.sqrt((A_flat**2).sum(dim=2) + eps)  # [B, C]
    if reduction == 'sum':
        return norms.sum()
    elif reduction == 'mean':
        return norms.mean()
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")


# 正交损失计算函数
def orthogonal_loss(a, E):

    M, H, W = a.shape
    device = a.device

    # 1. 计算端元间余弦相似度矩阵 C (M, M)
    E_normed = nn.functional.normalize(E, p=2, dim=1)  # (M, L)
    C = E_normed @ E_normed.t()  # (M, M)

    # 2. 构造 i != j 的 mask
    mask = 1.0 - torch.eye(M, device=device)  # (M, M)

    # 3. 计算各通道丰度在空间维度的内积矩阵 G
    #    先把空间展开： (M, H*W)
    a_flat = a.reshape(M, -1)  # (M, N) , N = H*W
    #    G[i,j] = sum_{h,w} a_i[h,w] * a_j[h,w]
    G = a_flat @ a_flat.t()  # (M, M)

    # 4. 加权求和，只保留 i != j
    #    sum_{i != j} G[i,j] * C[i,j]
    penalty = (G * C * mask).sum()

    # # 5. 归一化并加权
    # penalty = penalty / (H * W)

    # return loss_ortho
    return penalty

def convex_relaxed_orthogonal_loss(abundance_maps):
    """
    凸松弛后的正交损失函数：
    计算不同端元丰度图之间内积的绝对值之和（L1惩罚），
    并隐式约束每行L2范数不超过1（需在优化器中通过投影实现）。

    参数:
        abundance_maps (torch.Tensor): 丰度图张量，形状为 [C, W, H]

    返回:
        torch.Tensor: 凸松弛后的正交损失值（标量）
    """
    C, W, H = abundance_maps.shape
    A = abundance_maps.view(C, -1)  # [C, W*H]

    # 计算 Gram 矩阵
    G = torch.matmul(A, A.T)  # [C, C]

    # 提取非对角线元素并计算绝对值之和
    mask = ~torch.eye(C, dtype=torch.bool, device=A.device)
    off_diag = G[mask]  # 非对角元素
    loss = torch.sum(torch.abs(off_diag))

    return loss



# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse


# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


def l_half_norm(X):
    """
    计算二维数组 X 的 L1/2 范数，
    定义为 sum( |X_ij|^(1/2) )。
    """
    return torch.sum(torch.abs(X)**0.5)


# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input, endmember_number, col):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=1))
    abundance_input = torch.reshape(abundance_input.squeeze(0), (endmember_number, col, col))
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT, endmember_number):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT


# plot abundance
def plot_abundance(abundance_input, abundance_GT_input, endmember_number):
    plt.figure(figsize=(20, 10))
    for i in range(0, endmember_number):
        plt.subplot(2, endmember_number, i + 1)
        plt.imshow(abundance_input[i, :, :], cmap="jet")

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.imshow(abundance_GT_input[i, :, :], cmap="jet")
    plt.show()


# plot endmember
def plot_endmember(endmember_input, endmember_GT, endmember_number):
    plt.figure(figsize=(20, 5))
    for i in range(0, endmember_number):
        plt.subplot(1, endmember_number, i + 1)
        plt.plot(endmember_input[:, i], color="b")
        plt.plot(endmember_GT[:, i], color="r")
    plt.show()

def minmax_scale_per_row(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    对 A 的每一行，按 Min–Max 线性拉伸到对应的 B 的每一行范围。
    A, B 必须形状相同，且形状是 (3, N)。
    """
    if A.shape != B.shape:
        raise ValueError("A 和 B 必须形状相同")
    # 计算每行的 min/max
    min_A = A.min(axis=0, keepdims=True)   # (3,1)
    max_A = A.max(axis=0, keepdims=True)   # (3,1)
    min_B = B.min(axis=0, keepdims=True)
    max_B = B.max(axis=0, keepdims=True)

    # 防止某行全常数导致除零
    diff_A = max_A - min_A
    diff_A[diff_A == 0] = 1.0

    # 归一化到 [0,1]，再映射到 B 的行范围
    A_norm = (A - min_A) / diff_A
    return A_norm * (max_B - min_B) + min_B


# change the index of abundance and endmember true
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT, endmember_number):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[RMSE_index, :, :]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]
    return abundance_input, endmember_input, RMSE_abundance, SAD_endmember
