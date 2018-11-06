# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:26:03 2018

@author: zhangxinhui
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
from sklearn.pipeline import Pipeline
data_folder=os.path.join("D:\\","learning data\PythonDataMining-master\data",
                           "adult.data")
adult=pd.read_csv(data_folder,header=None,
                  names=["Age", "Work-Class", "fnlwgt",
                         "Education","Education-Num", "Marital-Status",
                         "Occupation","Relationship", "Race", "Sex", 
                         "Capital-gain","Capital-loss", "Hours-per-week", 
                         "Native-Country","Earnings-Raw"])
adult.dropna(how="all",inplace=True)
adult["LongHours"]=adult["Hours-per-week"]>40
X=adult[["Age","Education-Num","Capital-gain","Capital-loss",
        "Hours-per-week"]].values
Y=(adult["Earnings-Raw"]==">50K").values
transformer = SelectKBest(score_func=chi2, k=3)
xt_chi2=transformer.fit_transform(X,Y)
def multivariate_pearsonr(X, Y):
    scores,pvalues=[],[]
    for column in range(X.shape[1]):
        cur_score,cur_p=pearsonr(X[:,column],Y)
        pvalues.append(cur_p)
    return (np.array(scores),np.array[pvalues])
transformer=SelectKBest(score_func=multivariate_pearsonr,k=3)
xt_pearson=transformer.fit_transform(X,Y)


