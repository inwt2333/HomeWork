import numpy as np

# 加载数据
data = np.load('data_1000d.npy')  # shape: (300, 1000)

# 超参数
n_components = 100  # PCA 降维后的维度

# TODO: 实现你的算法
class MyKMeans:
    def __init__(self, n_clusters=7, max_iter=100, tol=1e-4, random_state=100):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        np.random.seed(random_state)

    def standardize(self, X):
        # 标准化数据
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std

    def PCA(self, X, n_components=100):
        # 使用 PCA 降维
        X_centered = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return np.dot(U[:, :n_components], np.diag(S[:n_components]))
    
    def compute_distance(self, X, centers):
        # 计算样本与中心的欧氏距离
        distances = np.zeros((X.shape[0], centers.shape[0]))
        for i, center in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
        return distances
    
    def initialize_centers(self, X):
        # 随机选取 k 个样本作为初始中心
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[random_indices]

    def assign_clusters(self, X, centers):
        # 计算每个样本与中心的距离，返回簇分配
        distances = self.compute_distance(X, centers)
        return np.argmin(distances, axis=1)

    def update_centers(self, X, labels):
        # 对每个簇取平均，更新中心
        new_centers = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            new_centers[i] = X[labels == i].mean(axis=0)
        return new_centers

    def fit(self, X, n_components=100, reduced=False):
        # 训练 KMeans 模型
        if not reduced:
            X = self.standardize(X)
            X = self.PCA(X, n_components=n_components)
        centers = self.initialize_centers(X)
        for iteration in range(self.max_iter):
            labels = self.assign_clusters(X, centers)
            new_centers = self.update_centers(X, labels)
            # 检查收敛
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers
        self.centers = centers
        self.labels_ = labels
        return self

    def predict(self, X, n_components=100):
        # 计算每个样本最近的簇中心
        standardized_X = self.standardize(X)
        reduced_X = self.PCA(standardized_X, n_components=n_components)
        return self.assign_clusters(reduced_X, self.centers)

# 训练模型
kmeans = MyKMeans(n_clusters=7)
kmeans.fit(data, n_components=n_components)

# 预测每个点属于哪个簇（0-6）
pred_labels = kmeans.predict(data, n_components=n_components)  # shape: (300,)

# 保存结果
np.save('pred_labels.npy', pred_labels)