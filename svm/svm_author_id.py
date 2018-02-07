#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#           number/total *...  number/total

# entrophy result = -0.5 * math.log(0.5, 2) - 0.5 * math.log(0.5, 2)
#inoformation gain = entrophy parent - entrophy children substracted sum

print 2.0/3.0
# entrophy = -2.0/3.0 * math.log(2.0/3.0, 2) - (1.0/3.0 * math.log(1.0/3.0, 2))

# result = 1.0 - (3.0/4.0)*0.9184 - (1.0/4.0)*0
# somehow it should be 0
entrophy = -0.5 * math.log(0.5, 2) - 0.5 * math.log(0.5, 2)
information_gain = 1.0 - (2.0/4.0)*entrophy - (2.0/4.0)*entrophy

print "entrophy = ", entrophy
print "information_gain = ", information_gain
exit()


#########################################################
### your code goes here ###
linear_svc = SVC(C=10000.0, kernel='rbf')
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
t0 = time()
linear_svc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = linear_svc.predict(features_test)

print "predicting time: ", round(time() - t1, 3), "s"

t2 = time()
acc = accuracy_score(labels_test, pred)
print "accuracy time:", round(time() -t2, 3), "s",  " accuracy: " , acc

print "10", pred[10]
print "26", pred[26]
print "50", pred[50]


print "chris", pred.tolist().count(1)


#########################################################


