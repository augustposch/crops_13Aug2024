import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from pandas.tseries.holiday import USFederalHolidayCalendar

from scipy import optimize

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import aposch_functions as aposch




# ## Prepare readout functions

# ### Classic readout function
# 
# - does 5-fold CV and prints confusion matrix, as well as precision, recall, f1, and the standard error of each score.

# In[40]:


# Code adapted from sklearn user guide 3.3.1.4

def all_inclusive_scorer(clf, X, y):
    print('Starting work on a new fold...')
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    pr = precision_score(y, y_pred)
    re = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1],
            'precision': pr,
            'recall': re,
            'f1': f1}


# In[41]:


def cv_readout(model, X, y):
    print('Performing 5-fold cross-validation...')
    #scores = cross_validate(model, X, y, cv=5, scoring=['precision','recall','f1','confusion_matrix'])
    raw = cross_validate(model, X, y, cv=5, scoring=all_inclusive_scorer)
    print('Aggregating the scores...')
    sum_cm = np.array([[sum(raw['test_tn']), sum(raw['test_fp'])],
                       [sum(raw['test_fn']), sum(raw['test_tp'])]])
    
    agg_scores = pd.DataFrame(index=['precision','recall','f1'],
                              columns=['mean','std'])
    agg_scores.at['precision','mean'] = np.mean(raw['test_precision'])
    agg_scores.at['precision','std'] = np.std(raw['test_precision'])
    agg_scores.at['recall','mean'] = np.mean(raw['test_recall'])
    agg_scores.at['recall','std'] = np.std(raw['test_recall'])
    agg_scores.at['f1','mean'] = np.mean(raw['test_f1'])
    agg_scores.at['f1','std'] = np.std(raw['test_f1'])
    
    print(sum_cm)
    print(agg_scores)
    


# ### Regression readout function

# Not finished yet. Currently is the same as the classification readout - I need to change the scores to be RMSE or similar.

# In[42]:


# Code adapted from sklearn user guide 3.3.1.4

def all_inclusive_scorer_r(reg, X, y):
    print('Starting work on a new fold...')
    y_pred = reg.predict(X)
    cm = confusion_matrix(y, y_pred)
    pr = precision_score(y, y_pred)
    re = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1],
            'precision': pr,
            'recall': re,
            'f1': f1}


# In[43]:


def cv_readout_r(model, X, y):
    print('Performing 5-fold cross-validation...')
    #scores = cross_validate(model, X, y, cv=5, scoring=['precision','recall','f1','confusion_matrix'])
    raw = cross_validate(model, X, y, cv=5, scoring=all_inclusive_scorer)
    print('Aggregating the scores...')
    sum_cm = np.array([[sum(raw['test_tn']), sum(raw['test_fp'])],
                       [sum(raw['test_fn']), sum(raw['test_tp'])]])
    
    agg_scores = pd.DataFrame(index=['precision','recall','f1'],
                              columns=['mean','std'])
    agg_scores.at['precision','mean'] = np.mean(raw['test_precision'])
    agg_scores.at['precision','std'] = np.std(raw['test_precision'])
    agg_scores.at['recall','mean'] = np.mean(raw['test_recall'])
    agg_scores.at['recall','std'] = np.std(raw['test_recall'])
    agg_scores.at['f1','mean'] = np.mean(raw['test_f1'])
    agg_scores.at['f1','std'] = np.std(raw['test_f1'])
    
    print(sum_cm)
    print(agg_scores)
    


# ### Special-case readout functions
# 
# - The 'r_to_c' function operates on a regression estimator, thresholds the predictions at 74, then scores it as classification. It does that in addition to providing a traditional RMSE score viewing the problem as regression.

# In[44]:

# Code adapted from sklearn user guide 3.3.1.4
# assumes input y is a regression target

def all_inclusive_scorer_r_to_c(reg, X, y):
    print('Starting work on a new fold...')
    # Regression metrics
    y_pred_reg = reg.predict(X)
    rmse = mean_squared_error(y, y_pred_reg, squared=False)
    
    # Classification metrics
    y_pred_clf = y_pred_reg > 74
    y_clf = y > 74 # go from regression to classification
    cm = confusion_matrix(y_clf, y_pred_clf)
    pr = precision_score(y_clf, y_pred_clf)
    re = recall_score(y_clf, y_pred_clf)
    f1 = f1_score(y_clf, y_pred_clf)
    return {'rmse': rmse,
            'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1],
            'precision': pr,
            'recall': re,
            'f1': f1}


# In[45]:


def cv_readout_r_to_c(model, X, y):
    print('Performing 5-fold cross-validation...')
    #scores = cross_validate(model, X, y, cv=5, scoring=['precision','recall','f1','confusion_matrix'])
    raw = cross_validate(model, X, y, cv=5, scoring=all_inclusive_scorer_r_to_c)
    print('Aggregating the scores...')
    sum_cm = np.array([[sum(raw['test_tn']), sum(raw['test_fp'])],
                       [sum(raw['test_fn']), sum(raw['test_tp'])]])
    
    agg_scores = pd.DataFrame(index=['precision','recall','f1','rmse'],
                              columns=['mean','std'])
    agg_scores.at['precision','mean'] = np.mean(raw['test_precision'])
    agg_scores.at['precision','std'] = np.std(raw['test_precision'])
    agg_scores.at['recall','mean'] = np.mean(raw['test_recall'])
    agg_scores.at['recall','std'] = np.std(raw['test_recall'])
    agg_scores.at['f1','mean'] = np.mean(raw['test_f1'])
    agg_scores.at['f1','std'] = np.std(raw['test_f1'])
    agg_scores.at['rmse','mean'] = np.mean(raw['test_rmse'])
    agg_scores.at['rmse','std'] = np.std(raw['test_rmse'])
    
    print(sum_cm)
    print(agg_scores)
    return sum_cm, agg_scores
    


# ### Funky scoring function

# **Note 8/8: I'm keeping these thoughts in here for now, but I've moved past some of these ideas after talking with Prof. Bogden.**
# 
# The goal here is to customize the scoring/error function to meet the needs of this project. For example, if the model predicts Crowded and there are actually 70 people in the vehicle, then that's not that bad. However, if the model predicts Crowded and there are acutally 50 people in the vehicle, that's pretty bad. (Likewise, if the model predicts NonCrowded and there are actually 80 people, that's not that bad; if the model predicts NonCrowded and there are actually 100 people, then that's pretty bad.
# 
# Finally, there's an asymmetric element of "a pleasant surprise is better than an unpleasant surprise". A rider such as Izzy would be fine with a pleasant surprise - predict Crowded but actual NonCrowded. But she would hate an unpleasant surprise - predict NonCrowded but then actually Crowded.
# - Deswegen ist es besser, oefter eine Crowded Berichtung zu geben.

# Approach A: Relatively simple. Treat this as classification, but allow for some slack in what's considered a correct prediction. If the prediction is within 5 of the Crowded threshold 74, then call it correct no matter what.
# - Implementing this through a classifier. Train the model using the True/False y targets. Evaluate the model on the gentler y targets where any passwithin between 69 and 79 gets marked correct.
#   - The above is tricky because it doesn't fall neatly into our cross-validation function. I *could* rewrite my cross-validation funciton to do it "by hand".
#     - How hard would it be to rewrite the crossvalidation function to be by hand? I need to do k-fold to get k=5 folds, then for each of the 5 folds I would need to fit a model on the training set, and get some scoring metrics out of the val set. I could score based on a different column than I trained on - it wouldn't be that hard. I might go forward with this approach...
#   - Simplified way to leverage the existing CV function. Idea: what if we train using *only* observations that are >79 `passwithin` or <=69 `passwithin`? This way, the model learns what a clearly uncrowded situation looks like and what a clearly crowded situation looks like. It's all right that we don't evaluate on the 69-79 observations, because we would mark them all correct anyway. I like this idea!
# - Implementing this through regression mindset. Need to make a regression prediction, then see how the prediction compares to the actual 'passwithin' value.

# ## Predictive models
# 
# These use one-hot encoding on the features. This is flawed but at least is a surefire way to help the model pick up on nonlinear effects of Hour, DOW, and Month.

# In[ ]:





# ### Create datasets



# In[156]:


# Datasets for classifier

def create_train_datasets(df05_train, feats, onehot=True, remove_borderline=True, sample_size=100000, problem='classification'):
    '''
    This function assumes we have df04 and df05 already. It's just a function for convenience.
    
    Output is an appropriate X_train and y_train.
    '''
    
    X_train = df05_train[feats]
    
    if problem == 'classification':
        y_train = df05_train['Crowded']
    
    if problem == 'regression':
        y_train = df05_train['passwithin']
        
    if remove_borderline==True:
        clear_obs = ~df05_train['passwithin'].between(70,79)
        X_train = X_train[clear_obs]
        y_train = y_train[clear_obs]
    
    if sample_size is not None:
        X_train = X_train.sample(sample_size, random_state=19)
        y_train = y_train.sample(sample_size, random_state=19)
    
    if onehot==True:
        enc = OneHotEncoder()
        X_train = enc.fit_transform(X_train)
    
    return X_train, y_train


## Create train dataset for time series prediction

def get_features_10x15min_for_row(pioneer, pioneer_ts, obs):
    '''
    Expects a dataframe of all passwithin observations, a Series of all the timestamps, and a row you're working with
    '''
    timestamp = pioneer.at[obs,'stop arrival time']
    idx_start = timestamp - pd.DateOffset(minutes=150)
    period_idx = pd.period_range(idx_start, freq='15min', periods=10)

    # get FMP masks
    FMP_masks = []  # FMP is a 15-minute period
    for i in range(10):
        boolean_membership = (pioneer_ts.index >= period_idx[i].start_time) & (pioneer_ts.index < period_idx[i].end_time)
        FMP_masks.append(boolean_membership)
    
    # calculate mean for each FMP and add it to the pioneer dataframe
    for i, FMP_mask in enumerate(FMP_masks):
        meanpass = pioneer.loc[FMP_mask,'passwithin'].mean()
        pioneer.at[obs,'FMP_'+str(i)] = meanpass
        
    return pioneer

def get_features_10x15min(pioneer, pioneer_ts):

    counter = 0
    for obs in pioneer.index:
        pioneer = get_features_10x15min_for_row(pioneer, pioneer_ts, obs)
        counter += 1
        if counter % 3000 == 0:
            print('did',counter,'rows so far')
    return pioneer


class PersistenceRegressor(BaseEstimator):
    def __init__(self):
        self.placeholder = None
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return X