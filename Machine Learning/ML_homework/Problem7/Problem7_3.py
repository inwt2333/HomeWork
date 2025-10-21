import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import datasets
from torchvision.transforms import ToTensor

# ===========================
# 1. 数据加载与预处理 (torchvision)
# ===========================
# 下载 MNIST 数据集
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())

# 将图像转换为 numpy 数组并展平
x_train_flat = np.array([img.numpy().reshape(-1) for img, label in mnist_train], dtype=np.float64)
y_train = np.array([label for img, label in mnist_train])

# 可选：为了内存限制，只保留每类前1000张图片
x_train_subsampled = []
y_train_subsampled = []
for digit in range(10):
    idx = np.where(y_train == digit)[0][:1000]
    x_train_subsampled.append(x_train_flat[idx])
    y_train_subsampled.append(y_train[idx])
x_train_flat = np.vstack(x_train_subsampled)
y_train = np.hstack(y_train_subsampled)

print(f"数据形状: {x_train_flat.shape}")  # (num_samples, 784)

# ===========================
# 2. 平均数字模板
# ===========================
digit1_images = x_train_flat[y_train == 1]
mean_digit1 = np.mean(digit1_images, axis=0)

plt.figure(figsize=(4, 4))
plt.imshow(mean_digit1.reshape(28, 28), cmap='gray')
plt.title("Average Digit-1 Template")
plt.axis('off')
plt.show()

# ===========================
# 3. PCA实现（标准协方差法 + Gram法）
# ===========================
def pca_standard(X, n_components=None):
    start_time = time.time()
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]
    elapsed = time.time() - start_time
    return eigvecs, eigvals, mean_X, elapsed

def pca_gram(X, n_components=None):
    start_time = time.time()
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    G = X_centered @ X_centered.T
    eigvals_small, eigvecs_small = np.linalg.eigh(G)
    idx = np.argsort(eigvals_small)[::-1]
    eigvals_small = eigvals_small[idx]
    eigvecs_small = eigvecs_small[:, idx]
    eigvecs = X_centered.T @ eigvecs_small
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
    if n_components is not None:
        eigvecs = eigvecs[:, :n_components]
        eigvals_small = eigvals_small[:n_components]
    elapsed = time.time() - start_time
    return eigvecs, eigvals_small, mean_X, elapsed

# ===========================
# 4. 前5个主成分
# ===========================
n_components = 5
eigvecs_std, eigvals_std, mean_X_std, time_std = pca_standard(x_train_flat, n_components)
eigvecs_gram, eigvals_gram, mean_X_gram, time_gram = pca_gram(x_train_flat, n_components)

print(f"Standard PCA time: {time_std:.4f}s")
print(f"Gram PCA time: {time_gram:.4f}s")

def plot_pcs(pcs, title):
    plt.figure(figsize=(12, 2))
    for i in range(pcs.shape[1]):
        plt.subplot(1, pcs.shape[1], i+1)
        plt.imshow(pcs[:, i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f'PC{i+1}')
    plt.suptitle(title)
    plt.show()

plot_pcs(eigvecs_std, "Top 5 PCs (Standard PCA)")
plot_pcs(eigvecs_gram, "Top 5 PCs (Gram PCA)")

# ===========================
# 5. 图像重构与MSE
# ===========================
def reconstruct_image(img, eigvecs, mean_X, n_list=[1, 2, 5, 10, 20]):
    img_centered = img - mean_X
    results = []
    for n in n_list:
        pcs = eigvecs[:, :n]
        coeffs = pcs.T @ img_centered
        recon = pcs @ coeffs + mean_X
        mse = np.linalg.norm(img - recon)**2
        results.append((recon, mse))
    return results

img_idx = 0
img_original = x_train_flat[img_idx]

recon_std = reconstruct_image(img_original, eigvecs_std, mean_X_std)
recon_gram = reconstruct_image(img_original, eigvecs_gram, mean_X_gram)

def plot_reconstructions(original, recon_list, method_name, n_list=[1,2,5,10,20]):
    plt.figure(figsize=(15, 3))
    plt.subplot(1, len(n_list)+1, 1)
    plt.imshow(original.reshape(28,28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    for i, (recon, mse) in enumerate(recon_list):
        plt.subplot(1, len(n_list)+1, i+2)
        plt.imshow(recon.reshape(28,28), cmap='gray')
        plt.title(f"n={n_list[i]}\nMSE={mse:.1f}")
        plt.axis('off')
    plt.suptitle(f"Reconstructions using {method_name}")
    plt.show()

plot_reconstructions(img_original, recon_std, "Standard PCA")
plot_reconstructions(img_original, recon_gram, "Gram PCA")
