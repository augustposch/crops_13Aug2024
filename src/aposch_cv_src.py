import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import cross_validate

averaging_strategy = 'weighted'

def crop_classif_scorer(clf, X, y, avg_sgy=averaging_strategy):
    
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    pr = precision_score(y, y_pred, average=avg_sgy, zero_division=0)
    re = recall_score(y, y_pred, average=avg_sgy)
    f1 = f1_score(y, y_pred, average=avg_sgy)
    scores_dict = {'precision': pr,
            'recall': re,
            'f1': f1,
            'cm_shape_0': cm.shape[0],
            'cm_shape_1': cm.shape[1]}
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            scores_dict['cm_'+str(i)+'_'+str(j)] = cm[i,j]
            
    print('Finished scoring that fold. Doing the next task...')
    return scores_dict

def cv_readout(model, X, y):
    print('Performing 5-fold cross-validation...')
    print('Doing the first fold...')
    raw = cross_validate(model, X, y, cv=5, scoring=crop_classif_scorer)
    #print('Here is the raw output:')
    #print(raw)
    print('Aggregating the scores...')
    
    cm_shape = raw['test_cm_shape_0'][0], raw['test_cm_shape_1'][0]
    sum_cm = np.full(cm_shape, -1)
    for i in range(cm_shape[0]):
        for j in range(cm_shape[1]):
            sum_cm[i,j] = np.sum(raw['test_cm_'+str(i)+'_'+str(j)])
    
    agg_scores = pd.DataFrame(index=['precision','recall','f1'],
                              columns=['mean','std'])
    agg_scores.at['precision','mean'] = np.mean(raw['test_precision'])
    agg_scores.at['precision','std'] = np.std(raw['test_precision'])
    agg_scores.at['recall','mean'] = np.mean(raw['test_recall'])
    agg_scores.at['recall','std'] = np.std(raw['test_recall'])
    agg_scores.at['f1','mean'] = np.mean(raw['test_f1'])
    agg_scores.at['f1','std'] = np.std(raw['test_f1'])
    
    print(agg_scores)
    return sum_cm, agg_scores