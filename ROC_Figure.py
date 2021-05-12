# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:24:25 2020

@author: Admin
"""
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class ROC_Figure(object):
    
    def __init__(self,Real_labels, Pred_labels):
        self.Real_labels = Real_labels
        self.Pred_labels = Pred_labels
        self.nClasses = Real_labels.shape[1]
        
    def getFigures(self):
        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(self.nClasses):
            fpr[i], tpr[i],_ = roc_curve(self.Real_labels[:,i],self.Pred_labels[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.figure()
            plt.plot(fpr[i], tpr[i], color=colors[i],lw=2, label='ROC curve (area = %0.4f)' % roc_auc[i]);
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.legend(loc="lower right")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Class# %2d' % i)
            plt.show()
                      
        
    def get_AllFiguresROC(self):
        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(self.nClasses):
            fpr[i], tpr[i],_ = roc_curve(self.Real_labels[:,i],self.Pred_labels[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i],lw=2, label='ROC curve (area = %0.4f)' % roc_auc[i]);
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class AUC')
        plt.show()
