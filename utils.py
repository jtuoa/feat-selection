from __future__ import division  # floating point division
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def split_data(x,y,split=0.8):
    trainsz = round(x.shape[0]*split)
    perm = np.random.permutation(x.shape[0])
    x = x[perm,...]
    y = y[perm]
    return (x[:trainsz],y[:trainsz]),(x[trainsz:],y[trainsz:])

#precision/recall functions from https://www.python-course.eu/confusion_matrix.php
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()

def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

def plot_cm(cm,lbls):
    fig=plt.figure()
    fig.add_subplot(111)
    sns.heatmap(cm,annot=True,square=True,cbar=False,fmt="d",cmap="YlOrRd_r")
    plt.xticks(np.arange(len(lbls)),lbls)
    plt.yticks(np.arange(len(lbls)),lbls)
    plt.title('All features')#Fisher 1000 features')
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.savefig('foo.png', bbox_inches='tight')
    plt.show()

def load_data(filename):
    data = np.load(filename).item()
    lbls = data['labels'].argmax(1)
    return (data['data'],lbls)


def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    for k in dict1:
        if k in dict2:
            dict1[k]=dict2[k]
