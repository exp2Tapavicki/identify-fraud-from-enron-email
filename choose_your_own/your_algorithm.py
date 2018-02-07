#!/usr/bin/python
from time import time

import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, tree, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


# clf = neighbors.KNeighborsClassifier(n_neighbors=180, weights='distance', algorithm='auto', leaf_size=50, p = 2, metric = 'minkowski', n_jobs = 8)
# print "start training"
# t0 = time()
# clf.fit(features_train, labels_train)
#
#
# print "training time:", round(time()-t0, 3), "s"
# t1 = time()
# pred = clf.predict(features_test)
# print "predicting time: ", round(time() - t1, 3), "s"
# acc = accuracy_score(labels_test, pred)
#
# print "k nearest neighbors acc = ", acc
# print "done"

# clf = RandomForestClassifier(
#     n_estimators=145,
#     criterion = 'entropy',
#     max_features ='auto',
#     # max_depth = None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.1, # checkout
#     max_leaf_nodes=5, # best one
#     min_impurity_decrease=0.0, # anything above doesn't help
#     bootstrap=True,
#     oob_score=False,
#     n_jobs=16,
#     verbose=0,
#     warm_start=False,
#     # class_weight='balanced',
#     random_state=0
# )
# print "start training"
# t0 = time()
# clf.fit(features_train, labels_train)
#
#
# print "training time:", round(time()-t0, 3), "s"
# t1 = time()
# pred = clf.predict(features_test)
# print "predicting time: ", round(time() - t1, 3), "s"
# acc = accuracy_score(labels_test, pred)
#
# print "RandomForestClassifier acc = ", acc
# print "done"
# 0.936


# clf = AdaBoostClassifier(
#     base_estimator=None,
#     n_estimators=50,
#     learning_rate=0.81,
#     algorithm='SAMME',
#     random_state=None
# )
# print "start training"
# t0 = time()
# clf.fit(features_train, labels_train)
#
#
# print "training time:", round(time()-t0, 3), "s"
# t1 = time()
# pred = clf.predict(features_test)
# print "predicting time: ", round(time() - t1, 3), "s"
# acc = accuracy_score(labels_test, pred)
#
# print "AdaBoostClassifier acc = ", acc
# print "done"
# 0.932

# clf = tree.DecisionTreeClassifier(
#     criterion='gini',
#     splitter='random',
#     max_depth=None,
#     min_samples_split=11,
#     min_samples_leaf=2,
#     min_weight_fraction_leaf=0.0,
#     max_features=None,
#     random_state=None,
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     class_weight=None,
#     presort=True
# )
# print "start training"
# t0 = time()
# clf.fit(features_train, labels_train)
#
# print "training time:", round(time() - t0, 3), "s"
# t1 = time()
# pred = clf.predict(features_test)
# print "predicting time: ", round(time() - t1, 3), "s"
# acc = accuracy_score(labels_test, pred)
#
# print "DecisionTreeClassifier acc = ", acc
# print "done"
# 0.936

clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    splitter='random',
    max_depth=None,
    min_samples_split=7,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    presort=True
)
print "start training"
t0 = time()
clf.fit(features_train, labels_train)

print "training time:", round(time() - t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "predicting time: ", round(time() - t1, 3), "s"
acc = accuracy_score(labels_test, pred)

print "SVC acc = ", acc
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print "precision is = ", precision
print "recall is = ", recall



print "done"

try:

    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
