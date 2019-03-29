%matplotlib inline
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


def create_dataset(n_samples=40000, weights=(0.01, 0.99), n_classes=2,
                   class_sep=0.8, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               #class_sep:乘以超立方体大小的因子。较大的值分散了簇/类，并使分类任务更容易。默认为1
                               class_sep=class_sep, 
                               random_state=44)

def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)

def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')




weights=((0.17, 0.83),(0.09, 0.91),(0.02, 0.98),(0.01, 0.99),(0.002, 0.998))
samples=(RandomUnderSampler(),NearMiss(version=1),NearMiss(version=2),NearMiss(version=3),TomekLinks(), \
        SMOTE(),BorderlineSMOTE(kind='borderline-1'),BorderlineSMOTE(kind='borderline-2'), \
        SVMSMOTE(),ADASYN(),SMOTETomek(smote=SMOTE(),tomek=TomekLinks()))
clfs=(SVC(gamma='auto'),DecisionTreeClassifier())

for weight in weights:
    for clf in clfs:
        for sample in samples:
            X, y = create_dataset(n_samples=40000, weights=weight,n_classes=2)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
            X_res,y_res=sample.fit_resample(X_train, y_train)
            clf=clf.fit(X_res,y_res)
            y_pre=clf.predict(X_test)
            print(clf.__class__.__name__+" "+sample.__class__.__name__)
            print("数据集比例为："+str(weight))
            fpr,tpr,threshold=roc_curve(y_test,clf.predict(X_test))
            rocauc=auc(fpr,tpr)#计算AUC
            print("ROC分数:{:.2f}".format(rocauc))
            print("*"*60)



samples=(RandomUnderSampler(),NearMiss(version=1),NearMiss(version=2),NearMiss(version=3),TomekLinks(), \
        SMOTE(),BorderlineSMOTE(kind='borderline-1'),BorderlineSMOTE(kind='borderline-2'), \
        SVMSMOTE(),ADASYN(),SMOTETomek(smote=SMOTE(),tomek=TomekLinks()),SMOTENC(categorical_features=[18,19]))
weights=((0.17, 0.83),(0.09, 0.91),(0.02, 0.98),(0.01, 0.99),(0.002, 0.998))
clfs=(SVC(gamma='auto'),DecisionTreeClassifier())

for clf in (SVC(gamma='auto'),DecisionTreeClassifier()):
	for weight in weights:
    for sample in samples:
	    X, y = make_classification(n_classes=2, class_sep=0.8,
	    weights=weight, n_informative=3, n_redundant=1, flip_y=0,
	    n_features=20, n_clusters_per_class=1, n_samples=40000, random_state=44)

	    #将最后两列设为名义变量
	    X[:, -2:] = RandomState(10).randint(0, 4, size=(40000, 2))
	    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
      X_res,y_res=sample.fit_resample(X_train, y_train)
	    clf=clf.fit(X_res,y_res)
	    y_pre=clf.predict(X_test)
		  print(clf.__class__.__name__+" "+sample.__class__.__name__)
	    print("数据集比例为："+str(weight))
	    fpr,tpr,threshold=roc_curve(y_test,clf.predict(X_test))
    	rocauc=auc(fpr,tpr)#计算AUC
    	print("ROC分数:{:.2f}".format(rocauc))
    	print("#"*60)


X,y=create_dataset(n_samples=50000,weights=(0.1,0.9),n_classes=2)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

samples=(RandomUnderSampler(),NearMiss(version=1),NearMiss(version=2),NearMiss(version=3),TomekLinks(), \
        SMOTE(),BorderlineSMOTE(kind='borderline-1'),BorderlineSMOTE(kind='borderline-2'), \
        SVMSMOTE(),ADASYN(),SMOTETomek(smote=SMOTE(),tomek=TomekLinks()))
clfs=(SVC(gamma='auto'),DecisionTreeClassifier())

for sample in samples:
    fig=plt.figure(figsize=(15,5))
    ax=fig.add_subplot(1,3,1)
    ax.set_title(sample.__class__.__name__)
    X_train,y_train=sample.fit_resample(X_train, y_train)
    plot_resampling(X_train, y_train, sample, ax)
    i=2
    for clf in clfs:
        clf=clf.fit(X_train,y_train)
        ax=fig.add_subplot(1,3,i)
        ax.set_title(clf.__class__.__name__)
        
        y_pre=clf.predict(X_test)
        plot_decision_function(X_test, y_pre, clf, ax)
        i+=1
    i+=1