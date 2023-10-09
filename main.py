# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    iris = pd.read_csv('Iris1.csv')
    # 查看信息
    iris.info()
    # # 设置颜色主题
    antV = ['#1890FF', '#2FC25B', '#FACC14']
    # 绘制 Violinplot 小钢琴图
    f,axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    sns.despine(left=True)
    sns.violinplot(x="Species", y='SepalLengthCm', data=iris, palette=antV, ax=axes[0, 0])
    sns.violinplot(x='Species', y='SepalWidthCm', data=iris, palette=antV, ax=axes[0, 1])
    sns.violinplot(x='Species', y='PetalLengthCm', data=iris, palette=antV, ax=axes[1, 0])
    sns.violinplot(x='Species', y='PetalWidthCm', data=iris, palette=antV, ax=axes[1, 1])
    plt.show()

    # 绘制 pointplot 点图
    f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    sns.despine(left=True)
    sns.pointplot(x='Species', y='SepalLengthCm', data=iris, color=antV[0], ax=axes[0, 0])
    sns.pointplot(x='Species', y='SepalWidthCm', data=iris, color=antV[0], ax=axes[0, 1])
    sns.pointplot(x='Species', y='PetalLengthCm', data=iris, color=antV[0], ax=axes[1, 0])
    sns.pointplot(x='Species', y='PetalWidthCm', data=iris, color=antV[0], ax=axes[1, 1])
    sns.pointplot()
    plt.show()


    # 生成各特征之间关系的矩阵图：
    sns.pairplot(data=iris, palette=antV, hue= 'Species')
    plt.show()



    # 排除非数值列
    numeric_columns = iris.select_dtypes(include=[np.number])
    # 计算相关性矩阵
    correlation_matrix = numeric_columns.corr()
    # 生成热图
    fig = sns.heatmap(correlation_matrix, annot=True, cmap='GnBu', linewidths=1, linecolor='k', square=True, mask=False,
                      vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True)

    plt.show()


    sns.lmplot(data=iris, x='SepalLengthCm', y='SepalWidthCm', palette=antV, hue='Species')
    plt.show()
    sns.lmplot(data=iris, x='PetalLengthCm', y='PetalWidthCm', palette=antV, hue='Species')
    plt.show()

    # 载入特征和标签集
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = iris['Species']
    # 对标签集进行编码
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # 查看编码后的标签值
    #print(y)
    # 接着，将数据集以
    # 7: 3
    # 的比例，拆分为训练数据和测试数据：
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=101)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # Support Vector Machine  支持向量机
    model = svm.SVC()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    print('The accuracy of the SVM is:{0}'.format(metrics.accuracy_score(prediction,test_y)))

    # Logistic Regression 逻辑回归
    model = LogisticRegression()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    print('The accuracy of the Logistic Regression is:{0}'.format(metrics.accuracy_score(prediction,test_y)))

    # 上面使用了数据集的所有特征，下面将分别使用花瓣和花萼的尺寸：
    petal = iris[['PetalLengthCm', 'PetalWidthCm', 'Species']]
    train_p, test_p = train_test_split(petal, test_size=0.3, random_state=0)
    train_x_p = train_p[['PetalWidthCm', 'PetalLengthCm']]
    train_y_p = train_p.Species
    test_x_p = test_p[['PetalWidthCm', 'PetalLengthCm']]
    test_y_p = test_p.Species
    sepal = iris[['SepalLengthCm', 'SepalWidthCm', 'Species']]
    train_s, test_s = train_test_split(sepal, test_size=0.3, random_state=0)
    train_x_s = train_s[['SepalWidthCm', 'SepalLengthCm']]
    train_y_s = train_s.Species
    test_x_s = test_s[['SepalWidthCm', 'SepalLengthCm']]
    test_y_s = test_s.Species

    # Support Vector Machine
    model = svm.SVC()
    model.fit(train_x_p, train_y_p)
    prediction = model.predict(test_x_p)
    print('The accuracy of the SVM using Petals is:{0}'.format(metrics.accuracy_score(prediction,test_y_p)))
    model.fit(train_x_s, train_y_s)
    prediction = model.predict(test_x_s)
    print('The accuracy of the SVM using Sepal is:{0}'.format(metrics.accuracy_score(prediction,test_y_s)))

    # Logistic Regression
    model = LogisticRegression()
    model.fit(train_x_p, train_y_p)
    prediction = model.predict(test_x_p)
    print('The accuracy of the Logistic Regression using Petals is:{0}'.format(metrics.accuracy_score(prediction,test_y_p)))
    model.fit(train_x_s, train_y_s)
    prediction = model.predict(test_x_s)
    print('The accuracy of the Logistic Regression using Sepals is:{0}'.format(metrics.accuracy_score(prediction,test_y_s)))

    # 从中不难看出，使用花瓣的尺寸来训练数据较花萼更准确。正如在探索性分
    # 析的热图中所看到的那样，花萼的宽度和长度之间的相关性非常低，而花瓣的宽
    # 度和长度之间的相关性非常高。