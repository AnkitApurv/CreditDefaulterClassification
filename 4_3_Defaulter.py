#!/usr/bin/env python
# coding: utf-8

# # Best Model for Defaulter

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
from sklearn.naive_bayes import GaussianNB

#classification result - statistical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
testX = ccdXdr
testY = ccdY

# ## 3.b. Oversampling

# In[13]:
"""

categoricalVars = {'LIMIT_BAL': False, 'SEX': True, 'EDUCATION': True, 'MARRIAGE': True, 'AGE': False, 
                   'PAY_1': True, 'PAY_2': True, 'PAY_3': True, 'PAY_4': True, 'PAY_5': True, 'PAY_6': True,
                   'BILL_AMT1': False, 'BILL_AMT2': False, 'BILL_AMT3': False, 'BILL_AMT4': False, 'BILL_AMT5': False, 'BILL_AMT6': False,
                   'PAY_AMT1': False, 'PAY_AMT2': False, 'PAY_AMT3': False, 'PAY_AMT4': False, 'PAY_AMT5': False, 'PAY_AMT6': False,
                   'PAY_MODE_SEVEREST': True, 'BILL_AMT_MEAN': False, 'PAY_AMT_MEAN': False}

def getSelectedCatBool(catVars, dfSelectedX):
    boolList = []
    for varName in dfSelectedX:
        if varName in list(catVars.keys()):
            boolList.append(catVars.get(varName))
    return boolList

trainXcat = getSelectedCatBool(categoricalVars, trainX.columns)


# In[14]:


oversampler = SMOTENC(categorical_features = trainXcat, sampling_strategy = 'minority', random_state = 44, n_jobs = -1)


# In[15]:


trainXoversampled, trainYoversampled = oversampler.fit_resample(trainX, trainY)

"""
# ## 4.a. Classification: GaussianNB

# In[16]:


#classifier = GaussianNB(priors=None, var_smoothing=1e-09)


# In[17]:


#classifier.fit(trainXoversampled, trainYoversampled.values.ravel())


# In[18]:


classifier = load('defaulter_classifier.joblib')


# ## 4.b. Classification Result - Statistical

# In[19]:


print(classifier.score(testX, testY))


# In[20]:


predictY = classifier.predict(testX)
print(classification_report(testY, predictY))


# In[21]:


cf = pandas.DataFrame(
    confusion_matrix(testY, predictY),
    columns=['Predicted | Not Defaulter', 'Defaulter'],
    index=['Correct | Not Defaulter', 'Defaulter'])

print(cf)
# 
