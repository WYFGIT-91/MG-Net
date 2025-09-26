'''
@Project ：MG-Net 
@File    ：Wavelet_Transform_Separation.py
@IDE     ：PyCharm 
@Author  ：王一梵
@Date    ：2025/9/26 10:38 
'''

import scipy.io as sio
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt


def wavelet_transform_separation(origin_img, Threshold=0.005):
    # 获取图像的波段数（通道数）
    num_bands = origin_img.shape[2]

    high_freq_reconstructed_image = np.zeros_like(origin_img, dtype=np.float32)
    low_freq_reconstructed_image = np.zeros_like(origin_img, dtype=np.float32)

    # 对每个通道进行傅里叶变换
    for band_index in range(num_bands):
        # 获取当前波段
        band = origin_img[:, :, band_index]

        # 对当前波段进行sym3小波变换
        coeffs = pywt.wavedec2(band, 'sym3', level=1, mode='smooth')

        cA, (cH, cV, cD) = coeffs

        # 应用软阈值处理，lambda即为阈值
        cH_thresh = pywt.threshold(cH, Threshold, mode='soft')
        cV_thresh = pywt.threshold(cV, Threshold, mode='soft')
        cD_thresh = pywt.threshold(cD, Threshold, mode='soft')

        # 重建低频特征
        low_freq_coeffs = (cA, (cH - cH_thresh, cV - cV_thresh, cD - cD_thresh))
        band_low_freq = pywt.waverec2(low_freq_coeffs, 'sym3', mode='smooth')
        low_freq_reconstructed_image[:, :, band_index] = band_low_freq[:band.shape[0], :band.shape[1]]

        # 重建高频特征
        high_freq_coeffs = (None, (cH_thresh, cV_thresh, cD_thresh))
        band_high_freq = pywt.waverec2(high_freq_coeffs, 'sym3', mode='smooth')
        high_freq_reconstructed_image[:, :, band_index] = band_high_freq[:band.shape[0], :band.shape[1]]

    return low_freq_reconstructed_image, high_freq_reconstructed_image
