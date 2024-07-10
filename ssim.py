import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os

# 파일 경로 수정
path1 = os.path.expanduser('C:\\capstone\\test4_origin.wav_mel_spectrogram.png')
path2 = os.path.expanduser('C:\\capstone\\test4_msdmpd_converted_generated.wav_mel_spectrogram.png')
path3 = os.path.expanduser('C:\\capstone\\test4_medmpdsnake_converted_generated.wav_mel_spectrogram.png')
path4 = os.path.expanduser('C:\\capstone\\test4_mrdmpd_converted_generated.wav_mel_spectrogram.png')

# 이미지 로드
img1 = img_as_float(imageio.imread(path1))
img2 = img_as_float(imageio.imread(path2))
img3 = img_as_float(imageio.imread(path3))
img4 = img_as_float(imageio.imread(path4))

# 이미지 크기 조정
min_rows = min(img1.shape[0], img2.shape[0], img3.shape[0], img4.shape[0])
min_cols = min(img1.shape[1], img2.shape[1], img3.shape[1], img4.shape[1])

img1 = img1[:min_rows, :min_cols]
img2 = img2[:min_rows, :min_cols]
img3 = img3[:min_rows, :min_cols]
img4 = img4[:min_rows, :min_cols]

# win_size 선택
win_size = min(3, min_rows, min_cols) if min(min_rows, min_cols) >= 3 else 1

# SSIM과 MSE 계산
def calculate_metrics(imgA, imgB):
    ssim_total = 0
    mse_total = 0
    for i in range(3):  # RGB 채널을 순회
        ssim_val = ssim(imgA[:,:,i], imgB[:,:,i], win_size=win_size, data_range=imgB[:,:,i].max() - imgB[:,:,i].min())
        mse_val = mean_squared_error(imgA[:,:,i], imgB[:,:,i])
        ssim_total += ssim_val
        mse_total += mse_val
    return ssim_total / 3, mse_total / 3  # 평균값 반환

ssim12, mse12 = calculate_metrics(img1, img2)
ssim13, mse13 = calculate_metrics(img1, img3)
ssim14, mse14 = calculate_metrics(img1, img4)


# 시각화
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img1)
ax[0].set_xlabel('Original Image')
ax[0].set_title('original')

ax[1].imshow(img2)
ax[1].set_xlabel(f'MSE: {mse12:.2f}, SSIM: {ssim12:.2f}')
ax[1].set_title('msdmpd')

ax[2].imshow(img3)
ax[2].set_xlabel(f'MSE: {mse13:.2f}, SSIM: {ssim13:.2f}')
ax[2].set_title('medmpdsnake')


ax[3].imshow(img4)
ax[3].set_xlabel(f'MSE: {mse14:.2f}, SSIM: {ssim14:.2f}')
ax[3].set_title('mrdmpd')

plt.tight_layout()
plt.show()