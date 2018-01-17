# Identity Fraud From Enron Email

## Project Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.

## Project Overview
In this project, you will play detective, and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

### Why this Project?
This project will teach you the end-to-end process of investigating data through a machine learning lens.

It will teach you how to extract/identify useful features that best represents your data, a few of the most commonly used machine learning algorithms today, and how to evaluate the performance of your machine learning algorithms.

### What will I learn?
By the end of the project, you will be able to:

Deal with an imperfect, real-world dataset
Validate a machine learning result using test data
Evaluate a machine learning result using quantitative metrics
Create, select and transform features compare the performance of machine learning algorithms
Tune machine learning algorithms for maximum performance
Communicate your machine learning algorithm results clearly
Why is this Important to my Career?
Machine Learning is a first-class ticket to the most exciting careers in data analysis today.

As data sources proliferate along with the computing power to process them, going straight to the data is one of the most straightforward ways to quickly gain insights and make predictions.

Machine learning brings together computer science and statistics to harness that predictive power.


## Resources Needed
You should have python and sklearn running on your computer, as well as the starter code (both python scripts and the Enron dataset) that you downloaded as part of the first mini-project in the Intro to Machine Learning course. The starter code can be found in the final_project directory of the codebase that you downloaded for use with the mini-projects. Some relevant files:

poi_id.py : starter code for the POI identifier, you will write your analysis here

final_project_dataset.pkl : the dataset for the project, more details below

tester.py : when you turn in your analysis for evaluation by a Udacity evaluator, you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py). The evaluator will then use this code to test your result, to make sure we see performance that’s similar to what you report. You don’t need to do anything with this code, but we provide it for transparency and for your reference.

emails_by_address : this directory contains many text files, each of which contains all the messages to or from a particular email address. It is for your reference, if you want to create more advanced features based on the details of the emails dataset.

## Steps to Success
We will provide you with starter code, that reads in the data, takes your features of choice, then puts them into a numpy array, which is the input form that most sklearn functions assume. Your job is to engineer the features, pick and tune an algorithm, test, and evaluate your identifier. Several of the mini-projects were designed with this final project in mind, so be on the lookout for ways to use the work you’ve already done.

The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)
email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)
POI label: [‘poi’] (boolean, represented as integer)
You are encouraged to make, transform or rescale new features from the starter features. If you do this, you should store the new feature to my_dataset, and if you use the new feature in the final algorithm, you should also add the feature name to my_feature_list, so your coach can access it during testing. For a concrete example of a new feature that you could add to the dataset, refer to the lesson on Feature Selection.


## Understanding the Dataset and Questions

> 1. Summarize for us the goal of this project and how machine learning is useful in trying to
accomplish it. As part of your answer, give some background on the dataset and how it
can be used to answer the project question. Were there any outliers in the data when
you got it, and how did you handle those? [relevant rubric items: “data exploration”,
“outlier investigation”]


The goal of this project is to see how machine learning can be applied as solution to real problem. Identify former Enron employee as POI/non POI (POI- Persons of Interest) using suitable machine learning algorithm.

### Dataset analysis:

* Total number of data points: 146
* Total number of POI(persons of interest):18
* Total number of non POI: 128
* Total number of features per person: 21

Percentage of persons who has email data =  60.1398601399 %

Percentage of existing data in data set by feature

* poi  =  100.0 %
* fraction_form_this_person_to_poi  =  100.0 %
* fraction_to_this_person_from_poi  =  100.0 %
* total_stock_value  =  87.4125874126 %
* total_payments  =  86.013986014 %
* restricted_stock  =  76.2237762238 %
* exercised_stock_options  =  70.6293706294 %
* expenses  =  65.7342657343 %
* salary  =  65.7342657343 %
* other  =  63.6363636364 %
* to_messages  =  60.1398601399 %
* from_poi_to_this_person  =  60.1398601399 %
* shared_receipt_with_poi  =  60.1398601399 %
* from_messages  =  60.1398601399 %
* from_this_person_to_poi  =  60.1398601399 %
* bonus  =  56.6433566434 %
* long_term_incentive  =  45.4545454545 %
* deferred_income  =  33.5664335664 %
* deferral_payments  =  26.5734265734 %
* restricted_stock_deferred  =  11.8881118881 %
* director_fees  =  11.1888111888 %
* loan_advances  =  2.0979020979 %

Person who don't have any feature value
* LOCKHART EUGENE E

Not person in our data 
* THE TRAVEL AGENCY IN THE PARK

There is 3 outliers that I have removed from data
* TOTAL (spreadsheet quirk)
* LOCKHART EUGENE E (doesn't have any feature value)
* THE TRAVEL AGENCY IN THE PARK (It is not person.)

## Optimize Feature Selection/Engineering

> 2\. What features did you end up using in your POI identifier, and what selection process did
you use to pick them? Did you have to do any scaling? Why or why not? As part of the
assignment, you should attempt to engineer your own feature that does not come
ready-made in the dataset -- explain what feature you tried to make, and the rationale
behind it. (You do not necessarily have to use it in the final analysis, only engineer and
test it.) In your feature selection step, if you used an algorithm like a decision tree,
please also give the feature importances of the features that you use, and if you used an
automated feature selection function like SelectKBest, please report the feature scores
and reasons for your choice of parameter values. \[relevant rubric items: “create new
features”, “intelligently select features”, “properly scale features”\]

I have used all features including 2 new created features (fraction_to_this_person_from_poi, fraction_form_this_person_to_poi). 
Since we have a lot of financial data(difference in numbers can be huge like the number itself) I have scaled features using MinMaxScaler().
After scaling I have used PCA for dimensionality reduction retaining 15 components. From this 15 components
I used SelectKBest to retain 10 components with best score. 

Since everyone has to have email data, just data is not available to us, I will put 0.3 as default value for persons
who don't have email data for new features.

fraction_to_this_person_from_poi = from_poi_to_this_person / from_messages

fraction_form_this_person_to_poi = from_this_person_to_poi / to_messages

SelectKBest.scores_ [  3.19504799  11.18847351   0.34750571   1.10765102   1.23383464
   2.64401092   0.0261739    0.57082264   8.49077675   0.0946169
   0.0251451    0.44986629   5.44671625   0.02889075   2.77271237]

After preparing of data is finished, I have used it with different algorithm. Scores are shown in table below:

Algorithm | Accuracy | Precision | Recall
---|---:|---:|---:
 GaussianNB | 0.82753 | 0.36927 | 0.41450 
 SVC | 0.88372 | 0.0 | 0.0 
 DecisionTreeClassifier | 0.78753 | 0.22813 | 0.24900  
 RandomForestClassifier | 0.85347 | 0.34954 | 0.11500 
 Lasso | 0.0 | 0.0 | 0.0 

In my final solution I have included 2 new features since results are better with them.

Accuracy | Precision | Recall
:---:|:---:|:---:
0.80707 | 0.31755 | 0.38900
0.82753 | 0.36927 | 0.41450

## Pick and Tune an Algorithm

> 3\. What algorithm did you end up using? What other one(s) did you try? How did model
performance differ between algorithms? \[relevant rubric item: “pick an algorithm”\]

Ideally we want to have high accuracy with high precision and recall. After five algorithm
(GaussianNB, SVC, DecisionTreeClassifier, RandomForestClassifier, Lasso)I have concluded 
that best algorithm would be Naive bayes. Since we don't have a lot of data like millions of records and 
our features having a lot of NaN value Naive bayes is best suited for current situation.

> 4\. What does it mean to tune the parameters of an algorithm, and what can happen if you
don’t do this well? How did you tune the parameters of your particular algorithm? What
parameters did you tune? (Some algorithms do not have parameters that you need to
tune -- if this is the case for the one you picked, identify and briefly explain how you
would have done it for the model that was not your final choice or a different model that
does utilize parameter tuning, e.g. a decision tree classifier). \[relevant rubric items:
“discuss parameter tuning”, “tune the algorithm”\] 

Tuning parameters of algorithm is a process in witch we are using all input parameters of 
algorithm that will impact the model in order to enable the algorithm to perform the best.
Of course best should be defined by what is important to us.

I used GridSearchCV to find it optimal parameter values.

#### SVC
* kernel = ['rbf', 'linear', 'poly']
* C = [0.1, 1, 10, 100, 1000, 10000]
* gamma = [1, 0.1, 0.01, 0.0001, 0.00001, 0.000001]
* random_state = [42]

#### DecisionTreeClassifier
* criterion = ['gini', 'entropy']
* splitter = ['best', 'random']
* max_depth = [1,10,50,80,100,150,200]
* min_samples_split = [2]
* min_samples_leaf = [1]
* min_weight_fraction_leaf = [0]
* max_features = ['auto', 'sqrt', 'log2', None]
* random_state = [42, None]
* max_leaf_nodes = [None]
* min_impurity_decrease = [0.0]
* class_weight = [None, 'balanced']
* presort=[False, True]  

#### RandomForestClassifier

* n_estimators = [10, 20, 50]
* criterion = ['gini', 'entropy']
* max_features = ['auto']
* max_depth = [1,10,50, None]
* min_samples_split = [2]
* min_samples_leaf = [1]
* min_weight_fraction_leaf = [0]
* max_leaf_nodes = [None]
* min_impurity_decrease = [0.0]
* n_jobs = [-1]
* random_state = [42, None]
* class_weight = [None, 'balanced']

#### Lasso

* alpha= [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
* fit_intercept = [True, False]
* normalize = [True, False]
* precompute = [True, False]
* copy_X = [True, False]
* tol = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
* warm_start = [True, False]
* positive = [True, False]
* random_state = [None, 42]
* selection = ['cyclic', 'random']

## Validate and Evaluate
   
> 5\.What is validation, and what’s a classic mistake you can make if you do it wrong? How
did you validate your analysis? \[relevant rubric items: “discuss validation”, “validation
strategy”\]

Validation is process of evaluating trained model. Model is tested with testing data set. Testing data
set is separate portion of the same data set from which the training set is derived. This way it gives
estimate of performance on independent data set and check of model overfitting. Main goal is to test
generalization ability of a trained model. Classic mistake could be overfitting a model. The overfitted
model performs great on training data but will fail predicting with new data.

I have done validation using sklearn.cross_validation on dataset with test_size = 0.3 witch
is 30% for testing and 70% for training. I validate analysis using test_classifier() method
who uses StratifiedKFold cross validation. In some cases I have used from sklearn.metrics 
accuracy_score, precision_score and recall_score.

> 6\. Give at least 2 evaluation metrics and your average performance for each of them.
Explain an interpretation of your metrics that says something human-understandable
about your algorithm’s performance. \[relevant rubric item: “usage of evaluation metrics”\]    

Evaluation metrics I have used are accuracy_score, precision_score and recall_score.

accuracy_score (Accuracy classification score)  = number of items in class labeled correctly/all items in that class 

- This function compute subset accuracy: the set of predicted true_positive must exactly match the corresponding in true_positive.

precision_score = true_positives/(true_positives+false_positives) 

- Percentage of correctly identified/classified POI, exactness or quality.  

recall_score = true_positives/(true_positives+false_negatives) 

- Recall measures classification of true positives over all cases that are actually positives, completeness or quantity.


I have used Naive Bayes and I got results below:

Accuracy = 0.82753

-  How close to actual (true) value we are.

Precision = 0.36927

- There is 36.92% of chance that my model will predict actual POI.

Recall = 0.41450

- There is 41.45% chance that my model will predict actual POI correctly.
 
