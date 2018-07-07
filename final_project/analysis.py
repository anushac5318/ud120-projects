#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pprint
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from numpy import log
from numpy import sqrt
from numpy import float64
from numpy import nan

from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.neural_network import BernoulliRBM

from sklearn.cluster import KMeans
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
                "salary",
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
                "bonus",
                "total_payments",
                "total_stock_value"
                ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

identified_outliers = ["TOTAL"]

for outlier in identified_outliers:
    data_dict.pop(outlier)


### Task 2: Remove outliers
financial_data = featureFormat(data_dict, target+features_financial)
print(target+features_financial)
print(financial_data)
email_data = featureFormat(data_dict, target+features_email)
pos=1
poi=['red' if each[0]==True else 'blue' for each in email_data ]
salary=[each[1] for each in financial_data ]
from_messages=[each[1] for each in email_data ]
for i in range(1, len(features_financial)):
    plt.subplot(5, 3, pos)
    pos+=1
    values = [each[i+1] for each in financial_data]
    plt.scatter(salary, values, c=poi)
    plt.xlabel(features_financial[0])
    plt.ylabel(features_financial[i])
    print financial_data[0]
plt.show()
pos=1
for i in range(1, len(features_email)):
    plt.subplot(3, 2, pos)
    pos+=1
    values = [each[i+1] for each in email_data]
    plt.scatter(from_messages, values, c=poi)
    plt.xlabel(features_email[0])
    plt.ylabel(features_email[i])
    print financial_data[0]
plt.show()



#print data_dict
#print poi_recs
#print non_poi_recs
total_recs=len(data_dict)
poi_recs={n:d  for n,d in data_dict.iteritems() if d['poi'] == True}
non_poi_recs={n:d  for n,d in data_dict.iteritems() if d['poi'] == False}
print "No of records",len(data_dict)
print "Poi records",len(poi_recs)
print "Poi records",len(non_poi_recs)
print data_dict.values()[0].keys()
print [(field,sum(data_dict[each][field] !='NaN' for each in data_dict)) for field in data_dict.values()[0].keys()]
personWiseCount= [(name,sum(each !='NaN' for each in value.values())) for (name,value) in data_dict.iteritems()]
personWiseCount= [(name,value) for (name,value) in personWiseCount if value <2]

print([(each,data_dict[each])for (each,x) in personWiseCount])

my_dataset = data_dict
for person in my_dataset.values():
    person['fraction_from_poi'] = 0
    person['fraction_to_poi'] = 0
    if float(person['from_messages']) > 0:
        person['fraction_to_poi'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])


    if float(person['to_messages']) > 0:
        person['fraction_from_poi'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])

features_email.extend(['fraction_from_poi', 'fraction_to_poi'])
email_data = featureFormat(data_dict, target+features_email)


fraction_from_poi = [ person['fraction_from_poi'] for person in my_dataset.values() if float(person['from_messages']) > 0 and float(person['to_messages']) > 0]
fraction_to_poi=[ person['fraction_to_poi'] for person in my_dataset.values() if float(person['from_messages']) > 0 and float(person['to_messages']) > 0]
poi=['red' if person['poi']==True else 'blue' for person in my_dataset.values() if float(person['from_messages']) > 0 and float(person['to_messages']) > 0 ]
plt.scatter(fraction_to_poi, fraction_from_poi, c=poi)
plt.xlabel('fraction_to_poi')
plt.ylabel('fraction_from_poi'fwefdf)
plt.show()


x=[person for person in my_dataset.values() if person['poi'] == True and person['fraction_from_poi']!=0]
print 110*"*"
print x