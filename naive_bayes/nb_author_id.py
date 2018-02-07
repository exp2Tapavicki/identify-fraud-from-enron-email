#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# gnb = GaussianNB()
# t0 = time()
# gnb_fitted = gnb.fit(features_train, labels_train)
# print "training time:", round(time()-t0, 3), "s"
#
# t1 = time()
# pred = gnb_fitted.predict(features_test)
# print "training time:", round(time()-t1, 3), "s"
#
# acc = accuracy_score(labels_test, pred)
# print(acc)

dt = tree.DecisionTreeClassifier(min_samples_split=40)
dt.fit(features_train, labels_train)
pred = dt.predict(features_test)
acc = accuracy_score(labels_test, pred)
print "accuracy is = ", acc

# length = len(features_train[0])
# print "length = ", length




#########################################################


