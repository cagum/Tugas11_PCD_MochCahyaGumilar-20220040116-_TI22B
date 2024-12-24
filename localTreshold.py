import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def localTreshold(image, block_size, c):
    # Tambahkan padding pada gambar
    imgPad = np.pad(image, pad_width=block_size // 2, mode='constant', constant_values=0)
    threshold = np.zeros_like(image)

    # Iterasi melalui setiap pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Tentukan area lokal
            local_area = imgPad[i:i + block_size, j:j + block_size]
            local_mean = np.mean(local_area)
            threshold[i, j] = 255 if image[i, j] > (local_mean - c) else 0

    return threshold

# Baca gambar
image = imageio.imread('D:\\Perkuliahan\\S5\\Pengolahan Citra Digital\\s11\\klaus.jpg', pilmode='F')
image_color = imageio.imread('D:\\Perkuliahan\\S5\\Pengolahan Citra Digital\\s11\\klaus.jpg', pilmode='RGB')

# Terapkan threshold lokal
result = localTreshold(image, 15, 10)

# Membuat mask untuk segmentasi
mask = (result == 255).astype(np.uint8)
segmented = image_color * mask[:, :, np.newaxis]

# Plot hasil
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Gambar Asli Berwarna')
plt.imshow(image_color)

plt.subplot(1, 3, 2)
plt.title('Threshold Lokal')
plt.imshow(result, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Segmentasi Berdasarkan Threshold')
plt.imshow(segmented)
plt.show()
