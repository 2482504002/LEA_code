from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载乳腺癌数据集
bcw = datasets.load_breast_cancer()

# 提取特征和标签
X = bcw.data
y = bcw.target
feature_indices_c = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29]
'''feature_indices_s = [18,19]
feature_indices_c1 = [1,3,5,7,9,11,13,15,17]
feature_indices_c = [i for i in range(30) if i not in feature_indices_s and i not in feature_indices_c1]'''
selected_features_c = X[:, feature_indices_c]
'''feature_indices_s = [18,19]
feature_indices_c1 = [1,3,5,7,9,11,13,15,17]
feature_indices_c2 = [2,4,6,8,10,12,14,16,20,22,24,26,28]
feature_indices_c = [i for i in range(30) if i not in feature_indices_s and i not in feature_indices_c1 and i not in feature_indices_c2]
selected_features_c = X[:, feature_indices_c]
selected_features_c1 = X[:, feature_indices_c1]
selected_features_c2 = X[:, feature_indices_c2]
selected_features_s = X[:, feature_indices_s]
selected_features_c = X[:, feature_indices_c]'''
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(selected_features_c, y, test_size=0.2, random_state=42)



# 由于乳腺癌数据集没有明确的标签，我们在这里不使用标签进行聚类

# 数据预处理：标准化特征
scaler = StandardScaler()
feature_indices=[i for i in range(30)]
accc=[]
for i in range(10):
    feature_indices_c=feature_indices[:(i+1)*3]
    selected_features_c = X[:, feature_indices_c]
    X_scaled = scaler.fit_transform(selected_features_c)

    # 使用K-Means算法进行聚类
    # 由于是二分类问题，我们可以选择聚类数目为2
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)

    # 获取聚类标签
    labels = kmeans.labels_

    kk=(labels==y)
    kkk=kk.sum()/len(y)
    if kkk<0.5:
        kkk=1-kkk
    accc.append(kkk)
print(accc)
print(f"i={i},acc={kkk/len(y)}")
# 可视化聚类结果（如果特征维度高于2，需要先降维）
# 这里我们使用PCA降维到2维空间
'''pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_scaled)
reduced_data_with_labels = reduced_data[:, :2]

# 绘制散点图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data_with_labels[:, 0], reduced_data_with_labels[:, 1], c=labels, cmap='viridis', s=50)
plt.title('K-Means Clustering on Breast Cancer Wisconsin Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.show()'''