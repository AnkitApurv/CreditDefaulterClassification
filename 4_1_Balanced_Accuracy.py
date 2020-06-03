#!/usr/bin/env python
# coding: utf-8

# # Best Model for Balanced_Accuracy

# ## 1.a. Import: Libraries

# In[1]:


#data organizing
import pandas #storage
import numpy as np #data-type conversion
from os import getcwd

#scaling and encoding
from sklearn.preprocessing import StandardScaler

#dimentionality reduction/feature selection
#from sklearn.feature_selection import SelectKBest #count of k best features chi2
#from sklearn.feature_selection import mutual_info_classif

#smote for imbalanced classes
#from imblearn.over_sampling import SMOTENC

#preprocessing - data splitting
#from sklearn.model_selection import train_test_split

#classifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier

#classification result - statistical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.exceptions import *

#model persistence
from joblib import load


# ## 1.b. Import: Dataset

# In[2]:


#dtype changed from int64 to int32 to save space and speed up computation, no data was lost
def cvDefPay(prediction):
    mapper = {0: False, 1: True}
    return mapper.get(prediction)

url = getcwd() + '\\default of credit card clients.xls'
ccd = pandas.read_excel(io = url,                         sheet_name='Data', header = 1, index_col = 0,                         dtype = {'LIMIT_BAL': np.int32, 'AGE': np.int32, 'BILL_AMT1': np.int32, 'BILL_AMT2': np.int32, 'BILL_AMT3': np.int32, 'BILL_AMT4': np.int32, 'BILL_AMT5': np.int32, 'BILL_AMT6': np.int32, 'PAY_AMT1': np.int32, 'PAY_AMT2': np.int32, 'PAY_AMT3': np.int32, 'PAY_AMT4': np.int32, 'PAY_AMT5': np.int32, 'PAY_AMT6': np.int32},                        converters = {'default payment next month': cvDefPay})


# In[3]:


ccd.rename(columns = {'PAY_0': 'PAY_1'}, inplace = True)
ccd.rename(columns = {'default payment next month': 'default_payment_next_month'}, inplace = True)


# ## 2.a Feature Engineering

# #### 1. PAY_1 to PAY_6

# In[4]:


ccdr = pandas.read_excel(io = url, 
                        sheet_name='Data', header = 1, index_col = 0)
ccdr.rename(columns = {'PAY_0': 'PAY_1'}, inplace = True)


# In[5]:


ccdrHistory = ccdr[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
ccdrHistoryMode = ccdrHistory.mode(axis = 'columns')
ccdrHistorySeverest = ccdrHistoryMode.apply(func = max, axis = 'columns')
ccd['PAY_MODE_SEVEREST'] = ccdrHistorySeverest


# #### 2. BILL_AMT1 to BILL_AMT6

# In[6]:


ccdSpent = ccd[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
ccd['BILL_AMT_MEAN'] = np.int32(ccdSpent.mean(axis = 'columns').round())


# #### 3. PAY_AMT1 to PAY_AMT6

# In[7]:


ccdSettled = ccd[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
ccd['PAY_AMT_MEAN'] = np.int32(ccdSettled.mean(axis = 'columns').round())


# ## 2.b. Normalization

# Scaling: Only to reduce the effect of very large continuous variables (in distance based esimators).
# 
# Normalization: Also reduce the effect of skewness in variables.
# 
# No need in Decision Trees

# In[8]:


varsToScale = ['LIMIT_BAL', 'AGE', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
               'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'BILL_AMT_MEAN', 'PAY_AMT_MEAN']
scaler = StandardScaler(copy = True)


# In[9]:


for var in varsToScale:
    ccd[var] = scaler.fit_transform(ccd[var].values.reshape(-1, 1))


# ## 2.c. Feature Selection

# In[10]:


ccdY = pandas.DataFrame(ccd['default_payment_next_month'])
ccdX = ccd.drop(['default_payment_next_month'], axis = 'columns')



# In[11]:


#featureFilter = SelectKBest(score_func = mutual_info_classif, k = np.int32(len(ccdX.columns) * 0.75))
#featureFilter.fit(X = ccdX, y = ccdY.values.ravel())
filteredColumnsIndices = [ 0,  2,  5,  6,  7,  8,  9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]

ccdXdr = ccdX.iloc[:, filteredColumnsIndices]

testX = ccdXdr
testY = ccdY

# ## 2.d. Encoding

# OneHotEncoding should be done after dimentionality reduction to ensure that one of the categories of a variable isn't dropped during feature selection, which could have been the case if OneHotEncoder was used before Feature Selection.
# 
# No need to run any encoder since:
# 
# 1. The dataset's relevant categorical variables are pre-encoded via OrdinalEncoder.
# 2. All of the categorical variables' categories have some difference in distribution in correlation with the target variable, so OneHotEncoder should not be used.

# ## 3.a. Data Splitting

# Data is split before oversampling to avoid synthetic datapoints in test dataset.
# 
# Test dataset is separated even though GridSearchCV uses Stratified K-Fold cross-validation so that model's accuracy can be tested independently.

# In[12]:


#trainX, testX, trainY, testY = train_test_split(ccdXdr, ccdY, test_size = 0.25, stratify = ccdY, random_state = 44)


# ## 4. Classification: StackingClassifier

# In[13]:
"""

clfRF = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight='balanced',
                               criterion='entropy', max_depth=9.5, max_features='auto',
                               max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                               oob_score=False, random_state=39, verbose=0, warm_start=False)

clfLG = LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=1000, multi_class='auto', n_jobs=-1, penalty='none',
                   random_state=44, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)

clfTrue = GaussianNB(priors=None, var_smoothing=1e-09)

clfFalse = SVC(cache_size = 500, max_iter = -1, random_state = 44, kernel = 'linear', C = 10,
               class_weight = {True: 1.25, False: 1.0})

classifier = StackingClassifier(estimators = [('clfRF', clfRF), 
                                              ('clfTrue', clfTrue),
                                              ('clfFalse', clfFalse)],
                              final_estimator = clfLG,
                              n_jobs = -1,
                              passthrough = True)

"""
# In[14]:


#classifier.fit(trainX, trainY.values.ravel())


# In[15]:


classifier = load('best_balanced_accuracy_classifier.joblib')


# ## 4.a. Classification Result - Statistical

# In[16]:


print(classifier.score(testX, testY))


# In[17]:


predictY = classifier.predict(testX)
print(classification_report(testY, predictY))


# In[18]:


cf = pandas.DataFrame(
    confusion_matrix(testY, predictY),
    columns=['Predicted | Not Defaulter', 'Defaulter'],
    index=['Correct | Not Defaulter', 'Defaulter'])

print(cf)
# 
