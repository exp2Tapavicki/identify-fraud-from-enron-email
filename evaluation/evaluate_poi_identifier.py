#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn import tree, cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score

sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# pred[4] = 0.0
# pred[11] = 0.0
# pred[19] = 0.0
# pred[21] = 0.0
# 22
# 24
# 25
# 27

acc = accuracy_score(labels_test, pred)
print "accuracy is = ", acc

precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print "precision is = ", precision
print "recall is = ", recall


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)

print "precision is = ", precision
print "recall is = ", recall

