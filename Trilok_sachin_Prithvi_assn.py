#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


#Reading in the CSV File
df = pd.read_csv('sample_data_intw.csv')


# In[3]:


#Dropping Unnecassary features
unn_attr = ['msisdn','maxamnt_loans90', 'fr_ma_rech30', 'fr_ma_rech90','cnt_da_rech30' ,'fr_da_rech30','cnt_da_rech90','medianamnt_loans90','medianamnt_loans30','fr_da_rech90','pcircle','pdate']
data = df.drop(unn_attr, axis=1)


# In[4]:



data = data.drop(df.columns[0], axis=1)
features = []
for col in data.columns:
    if col != 'label':
        features.append(col)
    


# In[5]:


#Converting the data into absolute values
data = data.abs()
data.describe()


# In[6]:


#Removing Outliers
from scipy import stats
z_scores = stats.zscore(data)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

#for feature in features:
#    q = data[feature].quantile(0.99)
#    data = data[data[feature] < q]

data['maxamnt_loans30'].values[data['maxamnt_loans30'] > 12] = 6.0

#data1.describe()
data.describe()


# In[7]:


#Histogram of features
data.hist(bins=50,figsize=(100,80))
plt.show()

get_ipython().run_line_magic('matplotlib', 'qt')


# In[8]:


x = np.linspace(1,10000,1000)
plt.plot(x,data['maxamnt_loans30'][:1000])
plt.ylabel(features[5])
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#Correlation Matrix
corr_matrix = data.corr()

sn.heatmap(corr_matrix, annot=False)


# In[10]:


#Correlation
for feature in features:
    print('{}'.format(feature), df['label'].corr(df[feature]))


# In[11]:


#Stratified Shuffling
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, StratifiedKFold, train_test_split, cross_val_score

#X = data.drop('label', axis=1)
y = data['label']

print(data['daily_decr30'].isnull().sum())
print(data.shape)

#train_set, test_set = train_test_split(X, test_size=0.2, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, y):
    strat_train_set = data.iloc[train_index]
    strat_test_set = data.iloc[test_index]
    print((y[train_index].sum())/len(y[train_index]), (y[test_index].sum())/len(y[test_index]))
    


print(strat_train_set['daily_decr30'].isnull().sum())
print(strat_train_set.shape)
print(strat_test_set['daily_decr30'].isnull().sum())
print(strat_test_set.shape)


# In[12]:


#Scatter Matrix to look for correlation between selected parameters
from pandas.plotting import scatter_matrix

attr1 = ['daily_decr30','daily_decr90','rental30','rental90']
attr2 = ['cnt_loans30','amnt_loans30','payback30','payback90']
scatter_matrix(data[attr1], figsize=(20,12))
scatter_matrix(data[attr2], figsize=(20,12))


# In[13]:


#Dropping Highly Correlated Features
drop_attr = ['daily_decr90','rental90','payback90','maxamnt_loans30','amnt_loans30']


strat_train_set = strat_train_set.drop(drop_attr, axis=1)
strat_test_set = strat_test_set.drop(drop_attr, axis=1)

for i in drop_attr: 
    try: 
        features.remove(i) 
    except ValueError: 
        pass
    
print(len(features))
print(strat_train_set.shape)


# In[14]:


#Preparing the Data for Training

X_train = strat_train_set.drop('label', axis=1)
y_train = strat_train_set['label'].copy()
X_test = strat_test_set.drop('label', axis=1)
y_test = strat_test_set['label'].copy()

#Scaling
from sklearn.preprocessing import MinMaxScaler,StandardScaler

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
X_train_tr = std_scaler.fit_transform(X_train)
X_test_tr = std_scaler.fit_transform(X_test)


# In[15]:


#Checking Imbalance
count_classes = pd.value_counts(y_train, sort = True)
count_classes.plot(kind = 'bar', rot=0)

plt.title("Label Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")

y_0 = y_train[y_train == 0]
y_1 = y_train[y_train==1]

outlier_fraction = len(y_0)/len(y_1)
state = np.random.RandomState(42)

print(y_0.shape,X_train.shape, y_1.shape)


# In[16]:


#Training with XGBClassifier

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

weights = np.linspace(0.07, 0.20, 8)
param_grid = dict(scale_pos_weight=weights)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    
#sample_pos_weight chosen to be 0.14 because it approx equal to minority class/ majority class
xgb_clf = XGBClassifier(scale_pos_weight=0.14)
xgb_clf.fit(X_train_tr, y_train)

#Classifiers for under and over sampled datasets
xgb_clf_dn = XGBClassifier()
xgb_clf_os = XGBClassifier()


# In[17]:


from sklearn.metrics import classification_report,accuracy_score

#function to evaluate the scores
def evaluate(classifier, X, y):
    scores = cross_val_score(classifier, X, y, scoring='roc_auc', cv=3, n_jobs=-1)
    print('Cross Val Score: ',scores)
    
    y_preds = classifier.predict(X)
    y_probas = classifier.predict_proba(X)

    print('Precision and Recall Scores')
    print(precision_score(y, y_preds), recall_score(y, y_preds))

    cm = confusion_matrix(y, y_preds, labels=[0,1])
    print('Confusion Matrix:')
    print(cm)
    print("Classification Report :")
    print(classification_report(y,y_preds))
    
evaluate(xgb_clf,X_test_tr, y_test)


# In[18]:


#ROC plot
def plot_roc(classifier, X, y): 
    y_pred = classifier.predict(X)
    fpr_for, tpr_for, threshold = roc_curve(y, y_pred)
    auc_for = auc(fpr_for, tpr_for)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(fpr_for, tpr_for, linestyle='-', label='(auc = %0.3f)' % auc_for)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('False Positive Rate -->')
    plt.ylabel('True Positive Rate -->')

    plt.legend()

    plt.show()
    
plot_roc(xgb_clf, X_test_tr, y_test)


# In[20]:


#Training the model with RandomForests

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


forest_clf = RandomForestClassifier(n_estimators=10)
forest_clf_dn = RandomForestClassifier(n_estimators=10)
forest_clf_os = RandomForestClassifier(n_estimators=10)


forest_clf.fit(X_train_tr, y_train)

evaluate(forest_clf, X_test_tr, y_test)


# In[21]:


plot_roc(forest_clf, X_test_tr, y_test)


# In[22]:


#Sampling
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler

#Under Sampling
us = RandomUnderSampler()
X_dn, y_dn = us.fit_sample(X_train_tr, y_train)
X_test_dn, y_test_dn = us.fit_sample(X_test_tr, y_test)

forest_clf_dn.fit(X_dn, y_dn)
xgb_clf_dn.fit(X_dn, y_dn)

#OverSampling
os = RandomOverSampler()
X_up, y_up = os.fit_sample(X_train_tr, y_train)
X_test_up, y_test_up = os.fit_sample(X_test_tr, y_test)

forest_clf_os.fit(X_up, y_up)
xgb_clf_os.fit(X_up, y_up)


# In[23]:


#Forest Undersampled
print('Forest Undersampled')
evaluate(forest_clf_dn, X_test_dn, y_test_dn)
plot_roc(forest_clf_dn, X_test_dn, y_test_dn)

#Forest Oversampled
print('Forest Oversampled')
evaluate(forest_clf_os, X_test_up, y_test_up)
plot_roc(forest_clf_os, X_test_up, y_test_up)

#XGB Undersampled
print('XGB Undersampled')
evaluate(xgb_clf_dn, X_test_dn, y_test_dn)
plot_roc(xgb_clf_dn, X_test_dn, y_test_dn)

#XGB Oversampled
print('XGB Oversampled')
evaluate(xgb_clf_os, X_test_up, y_test_up)
plot_roc(xgb_clf_os, X_test_up, y_test_up)


# In[24]:


#Feature Importance w.r.t xgb_clf
feature_importances = xgb_clf.feature_importances_
sorted(zip(feature_importances, features), reverse=True)


# In[25]:


#RandomSearch CV for xgb undersampled
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


random_search=RandomizedSearchCV(xgb_clf_dn,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_dn,y_dn)


# In[26]:


#Saving the best model on undersampled data
import pickle
best_model_dn = random_search.best_estimator_

filename = 'xgb_model.sav'
pickle.dump(best_model_dn, open(filename, 'wb'))


# In[27]:


#Loading and Evaluating
loaded_model_dn = pickle.load(open(filename, 'rb'))

evaluate(best_model_dn, X_test_dn, y_test_dn)
plot_roc(best_model_dn, X_test_dn, y_test_dn)


# In[ ]:




