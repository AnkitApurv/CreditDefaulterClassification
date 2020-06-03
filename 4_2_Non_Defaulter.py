#!/usr/bin/env python
# coding: utf-8

# # Best Model for Non_Defaulter

# # 1. Libraries

# In[1]:


#data organizing
import pandas #storage
import numpy as np #data-type conversion
from os import getcwd

#preprocessing
#from sklearn.model_selection import train_test_split #to split the data

#classifier
from sklearn.tree import DecisionTreeClassifier

#classification result - statistical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#classification result - graphical
from sklearn.tree import export_graphviz

#model persistence
from joblib import load


# # 2. Dataset - Importing

# In[2]:


#dtype changed from int64 to int32 to save space and speed up computation, no data was lost
def cvDefPay(prediction):
    mapper = {0: False, 1: True}
    return mapper.get(prediction)

url = getcwd() + '\\default of credit card clients.xls'
ccd = pandas.read_excel(io = url,                         sheet_name='Data', header = 1, index_col = 0,                         dtype = {'LIMIT_BAL': np.int32, 'AGE': np.int32, 'BILL_AMT1': np.int32, 'BILL_AMT2': np.int32, 'BILL_AMT3': np.int32, 'BILL_AMT4': np.int32, 'BILL_AMT5': np.int32, 'BILL_AMT6': np.int32, 'PAY_AMT1': np.int32, 'PAY_AMT2': np.int32, 'PAY_AMT3': np.int32, 'PAY_AMT4': np.int32, 'PAY_AMT5': np.int32, 'PAY_AMT6': np.int32},                         converters = {'default payment next month': cvDefPay})


# In[3]:


ccd.rename(columns = {'PAY_0': 'PAY_1'}, inplace = True)
ccd.rename(columns = {'default payment next month': 'default_payment_next_month'}, inplace = True)


# # 3. Splitting the dataset

# ## 3.a. Feature engineering

# In[4]:


ccdr = pandas.read_excel(io = url, 
                        sheet_name='Data', header = 1, index_col = 0)
ccdr.rename(columns = {'PAY_0': 'PAY_1'}, inplace = True)

ccdrPayHistory = ccdr[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

ccdrPayHistoryMode = ccdrPayHistory.mode(axis = 'columns')
ccdrPayHistorySeverest = ccdrPayHistoryMode.apply(func = max, axis = 'columns')

ccd['PAY_MODE_SEVEREST'] = ccdrPayHistorySeverest


# In[5]:


ccdSpent = ccd[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
ccd['BILL_AMT_MEAN'] = np.int32(ccdSpent.mean(axis = 'columns').round())


# In[6]:


ccdSettled = ccd[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
ccd['PAY_AMT_MEAN'] = np.int32(ccdSettled.mean(axis = 'columns').round())


# ## 3.b. Splitting the dataset

# In[7]:


ccdY = pandas.DataFrame(ccd['default_payment_next_month'])
ccdX = ccd.drop(['default_payment_next_month'], axis = 'columns')


# In[8]:


#trainX, testX, trainY, testY = train_test_split(ccdX, ccdY, test_size = 0.25, random_state = 44)
testX = ccdX
testY = ccdY

# # 4. Classification: DecisionTree

# In[9]:


#classifier = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_depth = len(ccdX.columns) * 0.25, #random_state = 39)


# In[10]:


#classifier.fit(trainX, trainY)


# In[11]:


classifier = load('non_defaulter_classifier.joblib')


# ## 4.a. Classification Result - Statistical

# In[12]:


print(classifier.score(testX, testY))


# In[13]:


predictY = classifier.predict(testX)
print(classification_report(testY, predictY))


# In[14]:


#http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/
cf = pandas.DataFrame(
    confusion_matrix(testY, predictY),
    columns=['Predicted | Not Defaulter', 'Defaulter'],
    index=['Correct | Not Defaulter', 'Defaulter'])

print(cf)

# ## 4.b. Classification Result - Graphical

# In[15]:


export_graphviz(classifier, out_file='02BenchmarkDecisionTree.dot', feature_names = ccdX.columns,
                class_names = ['True', 'False'], label = 'all', impurity = True,
                rounded = True, proportion = False, filled = True)#, precision = 2)

