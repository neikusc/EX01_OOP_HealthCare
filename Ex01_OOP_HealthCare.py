# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>


import numpy as np
import pandas as pd
import pylab as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# <codecell>

# ============================ #
# Part 1
# ============================ #
df = pd.read_csv("/home/neik/GitHub/EX01_OOP_HealthCare/data.csv")


df['sex'] = [1 if x=='F' else 0 for x in df['sex']]
train = df[['age','frailty_index','sex']]
target = df['death']
model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)
prob = model.fit(train,target).predict_proba(train)


# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(target, prob[:, 1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()


# ============================ #
# Part 2
# ============================ #
import datetime as dt
#
policies=pd.read_csv("/home/neik/Dropbox/JobSearch/Companies/clinicast/policies.csv")
policies.columns=['pid','start_date','end_date','dob','sex']


def ageDiff(dateTuple):
    dateVal, dob = list(dateTuple)
    dobYYYY, dobMM, dobDD = map(int, dob.split('-'))
    dateYYYY, dateMM, dateDD = map(int, dateVal.split('-'))
    dobDate = dt.date(dobYYYY, dobMM, dobDD)
    dateValDate = dt.date(dateYYYY, dateMM, dateDD)
    return (dateValDate - dobDate).days/365.25


policies['agePolicyStart'] = map(ageDiff, zip(list(policies['start_date']),list(policies['dob'])))
policies['agePolicyEnd'] = map(ageDiff, zip(list(policies['end_date']),list(policies['dob'])))
policies['policyLength'] = map(ageDiff, zip(list(policies['start_date']),list(policies['end_date'])))


categories=pd.read_csv("/home/neik/GitHub/EX01_OOP_HealthCare/categories.csv")
claims=pd.read_csv("/home/neik/GitHub/EX01_OOP_HealthCare/claims.csv")

code2CategoryDict=dict(zip(list(categories['code']),list(categories['category'])))
print 'Uniques:', code2CategoryDict


class Patient(object):
    def __init__(self,id,policyLength):
        self.id=id
        self.policyLength=policyLength
        self.greenCount=0
        self.orangeCount=0
        self.redCount=0
    
    def addCode(self,code):
        if code2CategoryDict[code]=='green':
            self.greenCount+=1
        
        elif code2CategoryDict[code]=='orange':
            self.orangeCount+=1
        
        elif code2CategoryDict[code]=='red':
            self.redCount+=1
    
    def total(self):
        self.totalCount=self.greenCount + self.orangeCount + self.redCount
        return
    
    def average(self):
        self.averageGreenClaims=self.greenCount/self.policyLength
        self.averageOrangeClaims=self.orangeCount/self.policyLength
        self.averageRedClaims=self.redCount/self.policyLength
        self.averageTotalClaims=self.totalCount/self.policyLength        


patientDictionary={}
for id , policyLength in zip(list(policies['pid']),list(policies['policyLength'])):
    patientDictionary[id]=Patient(id, policyLength)


for id, code in zip(list(claims['pid']),list(claims['code'])):
    patient=patientDictionary[id]
    patient.addCode(code)


for patient in patientDictionary.values():
    patient.total()
    patient.average()


greenAverageList=[patientDictionary[id].averageGreenClaims for id in policies['pid']]
orangeAverageList=[patientDictionary[id].averageOrangeClaims for id in policies['pid']]
redAverageList=[patientDictionary[id].averageRedClaims for id in policies['pid']]
totalAverageList=[patientDictionary[id].averageTotalClaims for id in policies['pid']]

policies['greenAverage']=greenAverageList
policies['orangeAverage']=orangeAverageList
policies['redAverage']=redAverageList
policies['totalAverage']=totalAverageList


outputFile=policies[['pid','agePolicyStart','agePolicyEnd','greenAverage','orangeAverage','redAverage','totalAverage']]
outputFile.to_csv('/home/neik/GitHub/EX01_OOP_HealthCare/output.csv')

