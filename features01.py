
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_score, recall_score
iris = pd.read_csv('Iris1.csv')
# 载入特征和标签集
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']
# 对标签集进行编码 标签编码将 3 种鸢尾花的品种名称转换为分类值（0, 1, 2）
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# print(y)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

print('原数据集中的特征数：\n', X.shape[1])

#单变量特征
SB = SelectKBest(chi2, k=2)
X_new=SB.fit_transform(X, y)

# 打印新数据集的特征数
print('单变量特征选择的特征数：\n', X_new.shape[1])

model = svm.SVC()

# 原数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('特征过滤前准确率：', acc)

# 过滤后的新数据集
X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.7, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(precision_score(test_y, y_pred, average='weighted'))
print(recall_score(test_y, y_pred, average='weighted'))
print('特征过滤后准确率：', acc)