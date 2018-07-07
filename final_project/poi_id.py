#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data, test_classifier

from numpy import log
from numpy import sqrt
from numpy import float64
from numpy import nan

from time import time
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression


from sklearn.cluster import KMeans




def create_classifiers():

    clf_list = []

    params_pca = {"pca__n_components": [2, 3,  5, 10, 15], "pca__whiten": [False]}

    #
    clf_naive = GaussianNB()
    params_naive = {}
    clf_list.append( (clf_naive, params_naive) )
    pca_clf_naive = GaussianNB()
    pca_params_naive={}
    pca_params_naive.update(params_pca)
    pca = PCA()
    clf_list.append(  (Pipeline([("pca", pca), ("naive", pca_clf_naive)]), pca_params_naive) )

    #
    clf_tree = DecisionTreeClassifier()
    params_tree = { "min_samples_split":[2, 5, 10, 20],
                    "criterion": ('gini', 'entropy'),
                    'random_state':[50]
                    }
    clf_list.append( (clf_tree, params_tree) )
    pca_clf_tree = DecisionTreeClassifier()
    pca_params_tree = {"tree__min_samples_split":[2, 5, 10, 20],
                    "tree__criterion": ('gini', 'entropy'),'tree__random_state':[50]}
    pca_params_tree.update(params_pca)
    pca = PCA()
    clf_list.append((Pipeline([("pca", pca), ("tree", pca_clf_tree)]), pca_params_tree))

    #
    clf_linearsvm = LinearSVC()
    params_linearsvm = {"C": [0.1, 1, 5, 10, 100],
                        "tol":[10**-1,  10**-3, 10**-5],
                        "class_weight":['balanced']

                        }
    clf_list.append( (clf_linearsvm, params_linearsvm) )
    pca_clf_linearsvm = LinearSVC()
    pca_params_linearsvm = {"svm__C": [0.1, 1, 5, 10, 100],
                        "svm__tol": [10**-1,  10**-3, 10**-5],
                        "svm__class_weight": ['balanced']

                        }
    pca_params_linearsvm.update(params_pca)
    pca = PCA()
    clf_list.append((Pipeline([("pca", pca), ("svm", pca_clf_linearsvm)]), pca_params_linearsvm))

    #
    clf_adaboost = AdaBoostClassifier()
    params_adaboost = { "n_estimators":[20, 50, 100],
                        'learning_rate': [0.4, 0.6, 1]}
    clf_list.append( (clf_adaboost, params_adaboost) )
    # pca_clf_adaboost = AdaBoostClassifier()
    # pca_params_adaboost = { "adaboost__n_estimators":[20,  50, 100],
    #                     'adaboost__learning_rate': [0.4, 0.6, 1]}
    # pca_params_adaboost.update(params_pca)
    # pca = PCA()
    # clf_list.append((Pipeline([("pca", pca), ("adaboost", pca_clf_adaboost)]), pca_params_adaboost))
    #
    clf_random_tree = RandomForestClassifier()
    params_random_tree = {  "n_estimators":[2, 3, 5,10,15],
                            "criterion": ('gini', 'entropy'),
                            'min_samples_split': [1, 2, 4], 'max_features': [1, 2, 3,'sqrt',5,10]
                            }
    clf_list.append( (clf_random_tree, params_random_tree) )
    pca_clf_random_tree  = RandomForestClassifier()
    pca_params_random_tree  = {  "random_tree__n_estimators":[2, 3, 5,10,15],
                            "random_tree__criterion": ('gini', 'entropy'),
                            'random_tree__min_samples_split': [1, 2, 4]
                            }
    pca_params_random_tree.update(params_pca)
    pca = PCA()
    clf_list.append((Pipeline([("pca", pca), ("random_tree", pca_clf_random_tree )]), pca_params_random_tree ))

    #
    clf_log = LogisticRegression()
    params_log = {  "C":[0.05, 0.5, 1, 10, 10**2,10**5,],
                    "tol":[10**-1, 10**-5, 10**-10],
                    "class_weight":['balanced'],
                    "penalty": ['l2', 'l1']
                    }
    clf_list.append( (clf_log, params_log) )

    pca_clf_log = LogisticRegression()
    pca_params_log = {  "log__C":[0.05, 0.5, 1, 10, 10**2,10**5,],
                    "log__tol":[10**-1, 10**-5, 10**-10],
                    "log__penalty":['l2','l1'],
                    "log__class_weight":['balanced']
                    }
    pca_params_log.update(params_pca)
    pca = PCA()
    clf_list.append((Pipeline([("pca", pca), ("log", pca_clf_log)]), pca_params_log))

    return clf_list
def create_classifiers_features():

    clf_list = []

    params_kbest = {"kbest__k": [1,2, 3,  5, 10, 15]}


    kbest_clf_naive = GaussianNB()
    kbest_params_naive={}
    kbest_params_naive.update(params_kbest)
    kbest = SelectKBest()
    clf_list.append(  (Pipeline([("kbest", kbest), ("naive", kbest_clf_naive)]), kbest_params_naive) )

    #

    kbest_clf_tree = DecisionTreeClassifier()
    kbest_params_tree = {"tree__min_samples_split":[2, 5, 10, 20],
                    "tree__criterion": ('gini', 'entropy'),'tree__random_state':[50]}
    kbest_params_tree.update(params_kbest)
    kbest = SelectKBest()
    clf_list.append((Pipeline([("kbest", kbest), ("tree", kbest_clf_tree)]), kbest_params_tree))

    #

    kbest_clf_linearsvm = LinearSVC()
    kbest_params_linearsvm = {"svm__C": [0.1, 1, 5, 10, 100],
                        "svm__tol": [10**-1,  10**-3, 10**-5],
                        "svm__class_weight": ['balanced']

                        }
    kbest_params_linearsvm.update(params_kbest)
    kbest = SelectKBest()
    clf_list.append((Pipeline([("kbest", kbest), ("svm", kbest_clf_linearsvm)]), kbest_params_linearsvm))

    #

    kbest_clf_adaboost = AdaBoostClassifier()
    kbest_params_adaboost = { "adaboost__n_estimators":[20,  50, 100],
                        'adaboost__learning_rate': [0.4, 0.6, 1]}
    kbest_params_adaboost.update(params_kbest)
    kbest = SelectKBest()
    clf_list.append((Pipeline([("kbest", kbest), ("adaboost", kbest_clf_adaboost)]), kbest_params_adaboost))


    kbest_clf_random_tree  = RandomForestClassifier()
    kbest_params_random_tree  = {  "random_tree__n_estimators":[2, 3, 5,10,15],
                            "random_tree__criterion": ('gini', 'entropy'),
                            'random_tree__min_samples_split': [1, 2, 4]
                            }
    kbest_params_random_tree.update(params_kbest)
    kbest = SelectKBest()
    clf_list.append((Pipeline([("kbest", kbest), ("random_tree", kbest_clf_random_tree )]), kbest_params_random_tree ))

    #


    kbest_clf_log = LogisticRegression()
    kbest_params_log = {  "log__C":[0.05, 0.5, 1, 10, 10**2,10**5,],
                    "log__tol":[10**-1, 10**-5, 10**-10],
                    "log__penalty":['l2','l1'],
                    "log__class_weight":['balanced']
                    }
    kbest_params_log.update(params_kbest)
    kbest = SelectKBest()
    clf_list.append((Pipeline([("kbest", kbest), ("log", kbest_clf_log)]), kbest_params_log))

    return clf_list




def optimize(clf, params, features, labels,scv):


    clf = GridSearchCV(clf, params, scoring="precision",cv=scv)
    clf = clf.fit(features, labels)
    clf = clf.best_estimator_


    return clf


def optimize_classifiers(clf_list, features, labels,cv):
    """
    Takes a list of tuples for classifiers and parameters, and returns
    a list of the best estimator optimized to it's given parameters.
    """
    import time
    best_estimators = []
    for clf, params in clf_list:
        start_time = time.time()
        print clf,params
        clf_optimized = optimize(clf, params, features, labels,cv)
        best_estimators.append( clf_optimized )
        print clf_optimized
        print("--- %s seconds ---" % (time.time() - start_time))

    return best_estimators






###############################################################################
# Quantitative evaluation of the model quality on the test set
def evaluate_classifier(clf, labels,features,cv, folds=1000):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        return[accuracy,precision,recall,f1,f2]
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."




### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target=["poi"]
features_email = [
                "from_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                "to_messages"
                ]


### Financial features might have underlying features of bribe money
features_financial = [
                "bonus",
                "deferral_payments",
                "deferred_income",
                "director_fees",
                "exercised_stock_options",
                "expenses",
                "loan_advances",
                "long_term_incentive",
                "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary",
                "total_payments",
                "total_stock_value"
                ]




features_list = target + features_email + features_financial
NAN_value = 'NaN'
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers
identified_outliers = ["TOTAL",'LOCKHART EUGENE E']

for outlier in identified_outliers:
    data_dict.pop(outlier)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset.values():
    person['fraction_from_poi'] = 0
    person['fraction_to_poi'] = 0
    if float(person['from_messages']) > 0:
        person['fraction_to_poi'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])
    if float(person['to_messages']) > 0:
        person['fraction_from_poi'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])

features_list.extend(['fraction_from_poi', 'fraction_to_poi'])


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#Provided to give you a starting point. Try a variety of classifiers.

import pprint
# pprint.pprint(features_list[0:])
# selector = SelectKBest(k=10)
#
# x = selector.fit(features, labels)
# f_list = zip(selector.get_support(), features_list[1:], selector.scores_)
# f_list = sorted(f_list, key=lambda x: x[2], reverse=True)
# print selector.get_support()
# print selector.scores_
# print "K-best features:",
# pprint.pprint(f_list)



# scv = StratifiedShuffleSplit(labels, n_iter=50,test_size=0.3, random_state = 42)
# clf_list = create_classifiers_features()
# clf_list = optimize_classifiers(clf_list, features, labels,scv)
# scv = StratifiedShuffleSplit(labels, n_iter=1000,test_size=0.3, random_state = 42)
# summary_list={}
# summary_list1={}
# import time
# for i, clf in enumerate(clf_list):
#     start_time = time.time()
#     result=evaluate_classifier(clf, labels,features,scv)
#     summary_list1[i]=result
#     summary_list[clf] = result
#     print clf,result
#     print("--- %s seconds ---" % (time.time() - start_time))
# ordered_list = sorted(summary_list.keys(), key=lambda k: summary_list[k][3], reverse=True)
# print [(key,summary_list[key]) for key in summary_list.keys() if summary_list[key][1]>0.3 and summary_list[key][2]>0.3]
# print ordered_list
# print "*"*100
# print summary_list
# print "*"*100
#
# clf = ordered_list[0]
# scores = summary_list[clf]
# print "Best classifier is ", clf
# print "With scores of  accuracy,recall, precision,f1,f2: ", scores
#
# scv = StratifiedShuffleSplit(labels, n_iter=50,test_size=0.3, random_state = 42)
# clf_list = create_classifiers()
# clf_list = optimize_classifiers(clf_list, features, labels,scv)
# scv = StratifiedShuffleSplit(labels, n_iter=1000,test_size=0.3, random_state = 42)
# summary_list={}
# summary_list1={}
# import time
# for i, clf in enumerate(clf_list):
#     start_time = time.time()
#     result=evaluate_classifier(clf, labels,features,scv)
#     summary_list1[i]=result
#     summary_list[clf] = result
#     print clf,result
#     print("--- %s seconds ---" % (time.time() - start_time))
#
# ordered_list = sorted(summary_list.keys(), key=lambda k: summary_list[k][3], reverse=True)
# print [(key,summary_list[key]) for key in summary_list.keys() if summary_list[key][1]>0.3 and summary_list[key][2]>0.3]
# print ordered_list
# print "*"*100
# print summary_list
# print "*"*100
#
# clf = ordered_list[0]
# scores = summary_list[clf]
# print "Best classifier is ", clf
# print "With scores of  accuracy,recall, precision,f1,f2: ", scores

clf= DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,max_features=None, max_leaf_nodes=None, min_samples_leaf=20,min_samples_split=20, min_weight_fraction_leaf=0.0,presort=False, random_state=50, splitter='best')
test_classifier(clf, my_dataset, features_list)
# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)