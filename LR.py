# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 01:53:13 2019

@author: Basant
"""
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import scipy.io as sio
f0=sio.loadmat('CWRU.mat')

xtrain=f0['training_inputs']
xtest=f0['test_inputs']
ytrain=f0['training_results']
ytest=f0['test_results']

xtrain=xtrain.T
ytrain=ytrain.T
xtest=xtest.T
ytest=ytest.T

x_train=xtrain
y_train=ytrain
x_test=xtest
y_test=ytest

from sklearn.svm import SVC
##svm_lin= SVC( kernel='linear', C =5).fit(x_train, y_train)  # for linear
#svm_rbf = SVC( kernel='rbf', C =90,gamma=.12).fit(x_train, y_train)  # for rbf
#
#accuracy=svm_rbf.score(x_test,y_test)
#accuracy
#from sklearn.svm import SVC
#svm_lin= SVC( kernel='rbf', C =85,gamma=0.12).fit(x_train, y_train)  # for linear
#accuracy = svm_lin.score(x_test, y_test)

from sklearn.linear_model import LogisticRegression
svm_lin = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(x_train, y_train)
accuracy = svm_lin.score(x_test,y_test) 


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
#    cv2.imwrite('newImage.png',fig)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
# plot cm

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

prediction = svm_lin.predict(x_test) 
y_score = prediction
fscore=(f1_score(y_test, y_score, average="macro"))   
precision=(precision_score(y_test, y_score, average="macro")) 
recall=(recall_score(y_test, y_score, average="macro"))  

np.set_printoptions(precision=2)
class_names=['right','failure']
class_names=np.array(class_names)
# Plot non-normalized confusion matrix
#plot_confusion_matrix(y_test, prediction, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, prediction, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

decision=svm_lin.decision_function(x_test)
print('decision of each test input',decision)
print('prediction of each test input',prediction)

print('all data by the model svm lin')
print('accuracy by SVM lin', accuracy)
print('precision', precision)
print('recall', recall)
print('fscore', fscore)


plt.show()
