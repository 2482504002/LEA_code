import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import re as re
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('D:\datasets\Give_me_some_credit\cs-training.csv')
test_df = pd.read_csv('D:\datasets\Give_me_some_credit\cs-test.csv')

print (train_df.info())
print(train_df.head(5))

train_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
test_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)

train_df.duplicated().value_counts()
test_df.duplicated().value_counts()
train_df.loc[train_df['age'] < 18]
train_df.loc[train_df['age'] == 0, 'age'] = train_df['age'].median()
working = train_df.loc[(train_df['age'] >= 18) & (train_df['age'] <= 60)]
senior = train_df.loc[(train_df['age'] > 60)]
working_income_mean = working['MonthlyIncome'].mean()
senior_income_mean = senior['MonthlyIncome'].mean()
print (working_income_mean)
print (senior_income_mean)
train_df['MonthlyIncome'] = train_df['MonthlyIncome'].replace(np.nan,train_df['MonthlyIncome'].mean())
train_df['NumberOfDependents'].fillna(train_df['NumberOfDependents'].median(), inplace=True)

X = train_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
y = train_df['SeriousDlqin2yrs']
W = test_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
z = test_df['SeriousDlqin2yrs']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=111)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

for i in range(1, 10):
    X_scaled1=X_scaled[:, 0:i]
    # 使用K-Means算法进行聚类
    # 由于是二分类问题，我们可以选择聚类数目为2
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled1)

    # 获取聚类标签
    labels = kmeans.labels_

    kk=(labels==y_train)
    kkk=kk.sum()
    print(f"i = {i}, acc = {kkk/len(y_train)}")
# 可视化聚类结果（如果特征维度高于2，需要先降维）
# 这里我们使用PCA降维到2维空间

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_scaled1)
reduced_data_with_labels = reduced_data[:, :2]

# 绘制散点图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data_with_labels[:, 0], reduced_data_with_labels[:, 1], c=labels, cmap='viridis', s=50)
plt.title('K-Means Clustering on Breast Cancer Wisconsin Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.show()