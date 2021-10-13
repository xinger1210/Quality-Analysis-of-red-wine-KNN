# -*- encoding = utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve
train_data=pd.read_csv("winequality-red.csv")
# print(train_data)

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.hist(train_data['quality'],bins=10)
# plt.show()

# f = plt.figure()
# f.set_figwidth(20)
# f.set_figheight(10)
# x=train_data['quality']
# plt.plot(x,train_data['fixed acidity'],'r',label='Fixed acidity')
# plt.plot(x,train_data['free sulfur dioxide'],'pink',label='free sulfur dioxide')
# plt.plot(x,train_data['residual sugar'],'maroon',label='residual sugar')
# plt.plot(x,train_data['total sulfur dioxide'],'lightseagreen',label='total sulfur dioxide')
# plt.plot(x,train_data['volatile acidity'],'b',label='Volatile acidity')
# plt.plot(x,train_data['citric acid'],'g',label='citric acid')
# plt.plot(x,train_data['pH'],'y',label='pH')
# plt.plot(x,train_data['alcohol'],'v',label='alcohol')
# plt.plot(x,train_data['chlorides'],'c',label='chlorides')
# plt.plot(x,train_data['sulphates'],'m',label='sulphates')
# plt.plot(x,train_data['density'],'k',label='density')
# plt.legend(loc=0)
# plt.figure()
# plt.show()
#
# plt.plot(x,train_data['density'],'g',label='density')
# plt.show()

train_data.drop_duplicates(inplace=True)

# print(train_data.isnull().sum())


"""
发现数据特征属性之间的相关性
"""

Corr=train_data.corr()
# print(Corr)
Corr_res=[]
# print(train_data.info())
for i in range(0,len(train_data.dtypes)):
    for j in range(0,len(train_data.dtypes)):
        values=Corr.iloc[i:i+1,j:j+1].values
        if values>0.8 and values !=1:
            Corr_res.append(Corr.columns[i])

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
# print(train_data.iloc[:,:-1])
# print(train_data)
train_data.iloc[:,:-1]=std.fit_transform(train_data.iloc[:,:-1])
# print(train_data.iloc[::,-1])
X=train_data.iloc[::-1]
Y=train_data.iloc[:,-1]
# print(x)
# print(Y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i !=y_test))


# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color="blue",
#          marker="o",
#          markersize=10,
#          markerfacecolor="red"
#          )
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()
classifier2=KNeighborsClassifier(n_neighbors=21,
                                metric="manhattan",p=2,
                                weights="uniform"
                                )
classifier2.fit(x_train,y_train)
y_pred1=classifier2.predict(x_train)
y_pred2=classifier2.predict(x_test)
# print(y_pred1)
# print(y_pred2)
from sklearn.metrics import accuracy_score
# print("训练数据集的精度评分：",accuracy_score(y_train,y_pred1))
# print("测试数据集的精度评分：",accuracy_score(y_test,y_pred2))

# plt.figure()
# plt.plot(y_test,'o',color = 'blue',label = 'Actual Values')
# plt.plot(y_pred2,color = 'red',label = 'Predicted values')
# plt.legend()
# plt.show()
# print(train_data["quality"].value_counts())
#如果大于6为1，否则为0
train_data['quality'] = np.where(train_data['quality'] > 6, 1, 0)
# print(train_data['quality'].value_counts())
X = train_data.drop(['quality'], axis = 1).values
y = train_data['quality'].values
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   stratify = y,
                                                   test_size = 0.3,
                                                   random_state = 1111)
print(X_train.shape)
k=range(1,50)
testing_accuracy=[]
training_accuracy=[]
score=0
for i in k:
    knn=KNeighborsClassifier(n_neighbors=i)
    pipe_knn=Pipeline([
        (
            "scale",MinMaxScaler()
        )
        ,(
            "knn",knn
        )
    ])
    pipe_knn.fit(X_train,y_train)
    y_pred_train=pipe_knn.predict(X_train)
    training_accuracy.append(accuracy_score(y_train,y_pred_train))

    y_pred_test = pipe_knn.predict(X_test)
    acc_score=accuracy_score(y_test, y_pred_test)
    testing_accuracy.append(acc_score)

    if score<acc_score:
        score=acc_score
        best_k=i



print("最好的准确性分数:", score,"最好best_k",best_k)


#
#
# k = range(1, 50, 2)
# testing_accuracy = []
# training_accuracy = []
# score = 0
#
# for i in k:
#     knn = KNeighborsClassifier(n_neighbors=i)
#     pipe_knn = Pipeline([('scale', MinMaxScaler()), ('knn', knn)])
#     pipe_knn.fit(X_train, y_train)
#
#     y_pred_train = pipe_knn.predict(X_train)
#     training_accuracy.append(accuracy_score(y_train, y_pred_train))
#
#     y_pred_test = pipe_knn.predict(X_test)
#     acc_score = accuracy_score(y_test, y_pred_test)
#     testing_accuracy.append(acc_score)
#
#     if score < acc_score:
#         score = acc_score
#         best_k = i
#
#
#
# print("最好的准确性分数:", score,"最好best_k",best_k)
#
#
#
