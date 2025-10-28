import numpy as np

# 加载数据
data = np.load('data_1000d.npy')  # shape: (300, 1000)

# 超参数
n_components = 50  # PCA 降维后的维度


# 实现你的算法
class MyGMM:
    def __init__(self, n_components = 100, n_clusters=7, max_iter=100, tol=1e-6, random_state=22):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        np.random.seed(random_state)

    def standardize(self, X):
        # 标准化数据
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # 防止除以零
        return (X - mean) / std

    def PCA(self, X):
        # 使用 PCA 降维
        X_centered = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return np.dot(U[:, :self.n_components], np.diag(S[:self.n_components]))
    
    def initialize_parameters(self, X):
        # 初始化 GMM 参数
        N, D = X.shape
        self.pi = np.ones(self.k) / self.k  # 每个簇的混合系数 π_j
        self.mu = X[np.random.choice(N, self.k, replace=False)]  # 从样本随机挑 k 个初始均值
        self.sigma = np.array([np.eye(D) for _ in range(self.k)])  # 初始化协方差矩阵为单位矩阵
        return N, D

    def gaussian_pdf(self, X, mu, sigma):
        D = X.shape[1]
        reg = 1e-6
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        norm_const = 1.0 / np.sqrt((2*np.pi)**D * det)
        diff = X - mu
        exp_term = np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))
        return norm_const * exp_term

    def expectation_step(self, X):
        N = X.shape[0]
        resp = np.zeros((N, self.k))
        for j in range(self.k):
            resp[:, j] = self.pi[j] * self.gaussian_pdf(X, self.mu[j], self.sigma[j])
        # 归一化
        row_sum = resp.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1e-10  # 防止除以零
        resp = resp / row_sum
        return resp
    
    def maximization_step(self, X, resp):
        N, D = X.shape
        Nk = resp.sum(axis=0) + 1e-10  # 防止除以零
        self.pi = Nk / N
        self.mu = (resp.T @ X) / Nk[:, np.newaxis]
        for j in range(self.k):
            diff = X - self.mu[j]
            weighted_cov = (resp[:, j][:, np.newaxis] * diff).T @ diff
            self.sigma[j] = weighted_cov / Nk[j] + 1e-6 * np.eye(D)  # 加小项防止奇异

    def log_loss(self, X):
        # 计算对数似然损失
        log_likelihood = np.sum(np.log(np.sum([
            self.pi[j] * self.gaussian_pdf(X, self.mu[j], self.sigma[j])
            for j in range(self.k)
        ], axis=0)))
        return log_likelihood

    def fit(self, X, n_components=100):
        # 训练 GMM 模型
        standardized_X = self.standardize(X)
        reduced_X = self.PCA(standardized_X)
        N, D = self.initialize_parameters(reduced_X)
        # GMM 训练的具体实现
        prev_log_likelihood = None
        for iteration in range(self.max_iter):
            resp = self.expectation_step(reduced_X)
            self.maximization_step(reduced_X, resp)
            log_likelihood = self.log_loss(reduced_X)
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

    def predict(self, X):
        # 预测每个样本的簇标签
        standardized_X = self.standardize(X)
        reduced_X = self.PCA(standardized_X)
        resp = self.expectation_step(reduced_X)
        return np.argmax(resp, axis=1)
        

# 预测每个点属于哪个簇（0-6）
gmm = MyGMM(n_components=n_components)
gmm.fit(data)
pred_labels = gmm.predict(data)  # shape: (300,)

# 保存结果
np.save('pred_labels.npy', pred_labels)