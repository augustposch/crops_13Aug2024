#!/usr/bin/env python
# coding: utf-8

# # *ml-4A-use-all-data.ipynb*
# 
# Version 4A: Reads in full, 100% train and 0.1% val datasets for each fold.
# 
# Uses Bagging classifier to handle all the 100% of data (52 million observations in each training fold)
# 
# The models we're interested in running include: Logistic Regression, Linear SVM, Nonlinear SVM.

# In[1]:


import numpy as np
import pandas as pd
import os
from timeit import default_timer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier


# In[2]:


def create_models_bank(architecture):
    models_bank = {}
    increment = 0
    if architecture in ('RF','ET','RFET','ETRF'):
        for n_estimators in [200, 500]:
            for max_features in [0.05, 0.1, 0.2, 0.4, 1.0]:
                for min_samples_split in [2, 4]:
                    for bootstrap in [False, True]:
                        for class_weight in [None, 'balanced']:

                            increment += 1

                            three_digit = str(increment).zfill(3)
                            models_bank[three_digit] = {
                                'n_estimators': n_estimators,
                                'max_features': max_features,
                                'bootstrap': bootstrap,
                                'min_samples_split': min_samples_split,
                                'class_weight': class_weight,
                                'n_jobs': -1
                            }

    if architecture in ('LR','PL'):
        # 50 regularized models
        for l1_ratio in [0, 0.1, 0.5, 0.9, 1]:
            for C in [1, 10**-2, 10**-4, 10**-6, 10**-8]:
                for class_weight in [None, 'balanced']:
                    increment += 1
                    three_digit = str(increment).zfill(3)
                    models_bank[three_digit] = {
                        'solver': 'saga',
                        'penalty': 'elasticnet',
                        'l1_ratio': l1_ratio,
                        'C': C,
                        'class_weight': class_weight,
                        'max_iter': 100000,
                        'random_state': 19
                    }
        # 2 nonregularized models
        for class_weight in [None, 'balanced']:
            increment += 1
            three_digit = str(increment).zfill(3)
            models_bank[three_digit] = {
                'solver': 'saga',
                'penalty': 'none',
                'class_weight': class_weight,
                'max_iter': 20000,
                'random_state': 19
            }
            
    if architecture in ('LS'):
        # 30 linear svms
        for dual_penalty_loss in [(False,'l1','squared_hinge'),
                             (False,'l2','squared_hinge'),
                             (True,'l2','hinge')]:
            for C in [1, 10**-2, 10**-4, 10**-6, 10**-8]:
                for class_weight in [None, 'balanced']:
                    increment += 1
                    three_digit = str(increment).zfill(3)
                    models_bank[three_digit] = {
                        'dual': dual_penalty_loss[0],
                        'penalty': dual_penalty_loss[1],
                        'loss': dual_penalty_loss[2],
                        'C': C,
                        'class_weight': class_weight,
                        'max_iter': 100000,
                    }
                    
    if architecture in ('SV'):
        # 150 nonlinear svms
        for kernel in ['rbf','poly','sigmoid']:
            for C in [1, 10**-2, 10**-4, 10**-6, 10**-8]:
                for gamma in [100, 10, 1, 10**-1, 10**-2]:
                    for class_weight in [None, 'balanced']:
                        increment += 1
                        three_digit = str(increment).zfill(3)
                        models_bank[three_digit] = {
                                     'kernel': kernel,
                                     'C': C,
                                     'gamma': gamma,
                                     'degree': 3,
                                     'coef0': 0,
                                     'class_weight': class_weight,
                                     'max_iter': -1,
                                                }
                        
    if architecture in ('NB'):
        # 7 Naive Bayes
        for var_smoothing in [10**-6, 10**-7, 10**-8,
                              10**-9, 10**-10, 10**-11, 0]:
            increment += 1
            three_digit = str(increment).zfill(3)
            models_bank[three_digit] = {
                         'var_smoothing': var_smoothing
                                    }
            
    if architecture in ('KN','PK'):
        step1 = [int(np.round(1.4**x)) for x in range(7,21)]
        step2 = [x+1 if x%2==0 else x for x in step1]
        k_list = [1,3,5,7,9] + step2
        # 152 models
        for n_neighbors in k_list:
            for weights in ['uniform','distance']:
                for metric in ['manhattan','euclidean','chebyshev','canberra']:
                    increment += 1
                    three_digit = str(increment).zfill(3)
                    models_bank[three_digit] = {
                        'n_neighbors': n_neighbors,
                        'weights': weights,
                        'metric': metric
                    }
         
    if architecture in ('NP'): # "Neural: Perceptron"
        # 36 neural perceptrons
        for hidden_layer_sizes in [(100,), (100,100,100)]:
            for alpha in [1, 10**-2, 10**-4, 10**-6, 10**-8, 10**-10]:
                for activation in ['relu', 'logistic', 'tanh']:
                    increment += 1
                    three_digit = str(increment).zfill(3)
                    models_bank[three_digit] = {'solver': 'adam',
                        'hidden_layer_sizes': hidden_layer_sizes, 
                        'activation': activation,
                        'alpha': alpha,
                        'learning_rate_init': 0.001,
                        'random_state': 19,
                        'n_iter_no_change': 10,
                        'max_iter': 2000}
    
    return models_bank


# In[ ]:





# In[3]:


def grab_premade_X_y_train_val(training_sample_size,validation_sample_size,tile,
                     val_year,scheme_name,crop_of_interest_id,in_season):

    loc = f'../data/premade_{training_sample_size}_{validation_sample_size}'
    strings = []
    for arg in [tile,val_year,scheme_name,crop_of_interest_id,in_season]:
        strings.append(f'{arg}')
    most_of_name = '_'.join(strings) 
    
    Xy_trainval= ['X_train', 'X_val', 'y_train', 'y_val']
    
    d = {}
    for spec in Xy_trainval:
        d[spec] = np.load(f'{loc}/{most_of_name}_{spec}.npy')
    
    return d['X_train'], d['X_val'], d['y_train'], d['y_val']


# In[4]:


def fit_predict_report(model_name,
                      model,
                      training_sample_size,
                      validation_sample_size,
                      tile,
                      years,
                      scheme_name,
                      crop_of_interest_id,
                      in_season,
                      from_premade=True
                      ):
    
    # produce csv_name
    exempt = ['years', 'model']
    param_value_strings = [f'{model_name}',
                      f'{training_sample_size}',
                      f'{validation_sample_size}',
                      f'{tile}',
                      f'{scheme_name}',
                      f'{crop_of_interest_id}',
                      f'{in_season}']
    csv_name = '_'.join(param_value_strings) +'.csv'

    # check whether previously run and, if so, end the effort
    if csv_name in os.listdir('../data/results/'):
        return 'If you see this, the specified model was previously run.'

    print(f'-- Process for {csv_name} --')
    
    # below is actually fitting and predicting and reporting
    
    conf = []

    for val_year in years:
        print('Starting a fold...')
        print('> Assembling the datasets')
        # NEED: X_train, y_train, X_val, y_val
        if from_premade==True: 
            X_train, X_val, y_train, y_val = grab_premade_X_y_train_val(training_sample_size,
                    validation_sample_size,tile,val_year,scheme_name,
                    crop_of_interest_id,in_season)
        
        if from_premade!=True:
            return 'This function in this notebook is only for from_premade=True'
        
        print('> Fitting the model on the training set')
        model.fit(X_train, y_train)
        print('> Predicting on the validation set')
        pred = model.predict(X_val)

        print('> Recording performance metrics')
        act = y_val
        ActPred_00 = sum((act==0) & (pred==0))
        ActPred_01 = sum((act==0) & (pred==1))
        ActPred_10 = sum((act==1) & (pred==0))
        ActPred_11 = sum((act==1) & (pred==1))
        conf_1yr = [ActPred_00, ActPred_01, ActPred_10, ActPred_11]

        conf.append(conf_1yr)
        print('Finished a fold.')

    carr = np.array(conf)

    carr = np.row_stack([carr,np.full((2,4),-1)])

    # above we added the totals row
    # now we need to add the columns for precision and recall

    # create dataframe
    cdf = pd.DataFrame(data = carr,
                      index = [f'ValYear{yr}' for yr in years]+['Mean','StdE'],
                      columns = ['ActPred_00', 'ActPred_01', 
                                 'ActPred_10', 'ActPred_11']
                      )

    cdf['Precision'] = cdf.ActPred_11 / (cdf.ActPred_01 + cdf.ActPred_11)
    cdf['Recall'] = cdf.ActPred_11 / (cdf.ActPred_10 + cdf.ActPred_11)
    cdf['F1'] = 2*cdf.Precision*cdf.Recall / (cdf.Precision + cdf.Recall)
    for col in ['Precision','Recall','F1']:
        cdf.at['Mean',col] = np.mean(cdf.loc[:'ValYear2022',col])
        cdf.at['StdE',col] = np.std(cdf.loc[:'ValYear2022',col])
    
    
    param_strings = [f'# model_name: {model_name}',
                     f'# model: {model}',
                      f'# training_sample_size: {training_sample_size}',
                      f'# validation_sample_size: {validation_sample_size}',
                      f'# tile: {tile}',
                      f'# scheme_name: {scheme_name}',
                      f'# crop_of_interest_id: {crop_of_interest_id}',
                      f'# in_season: {in_season}']
    comment = '\n'.join(param_strings) + '\n' 
    with open(f'../data/results/{csv_name}', 'a') as f:
        f.write(comment)
        cdf.to_csv(f)
    
    print(f'Find results in ../data/results/{csv_name}')
    
    return f'Find results in ../data/results/{csv_name}'


# In[5]:


def return_model_object(model_name,
                        bagged,
                        n_estimators=1000,
                        sample_size=0.001,
                        bootstrap=True):
    architecture = model_name[:2]
    three_digit = model_name[2:]
    models_bank = create_models_bank(architecture)
    hp = models_bank[three_digit] # hyperparameters
        
    if architecture in ('RF'):
        model = RandomForestClassifier(**hp)
        
    if architecture in ('ET'):
        model = ExtraTreesClassifier(**hp)
        
    if architecture in ('LR'):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(**hp)
                              )
        
    if architecture in ('PL'):
        model = make_pipeline(StandardScaler(),
                              PCA(0.9),
                              LogisticRegression(**hp)
                              )
       
    if architecture in ('LS'):
        model = make_pipeline(StandardScaler(),
                              LinearSVC(**hp)
                             )
        
    if architecture in ('SV'):
        model = make_pipeline(StandardScaler(),
                              SVC(**hp)
                             )
       
    if architecture in ('NB'):
        model = GaussianNB(**hp)
       
    if architecture in ('KN'):
        model = make_pipeline(StandardScaler(),
                              KNeighborsClassifier(**hp)
                             )
        
    if architecture in ('PK'):
        model = make_pipeline(StandardScaler(),
                              PCA(0.9),
                              KNeighborsClassifier(**hp)
                             )
       
    if architecture in ('NP'):
        model = make_pipeline(StandardScaler(),
                       MLPClassifier(**hp)
                      )
       
    if not bagged:
        return model
    
    if bagged:
        return BaggingClassifier(base_estimator=model, # choose which model
                                n_estimators=n_estimators, # train many estimators
                                max_samples=sample_size, # train each estimator on small sample
                                max_features=1.0, # use all the features
                                bootstrap=bootstrap, # draw samples with replacement
                                bootstrap_features=False,
                                n_jobs=-1,
                                random_state=19)
    


# ### Run models

# In[ ]:


start = default_timer()

model_names = ['LR036','LS021','SV009','SV088','SV130',
               'SV085','SV059','LS001','LR043']  ## SPECIFY HERE
training_sample_size = 1.0
validation_sample_size = 0.001

for tile_coiid in [('10SFH',75),('15TVG',1)]:
    for scheme_name in ['14day','5day']:
        for in_season in [160]:
            for model_name in model_names:

                model = return_model_object(model_name,
                                bagged=True,
                                n_estimators=100,
                                sample_size=0.001,
                                bootstrap=True)
 
                p = {

                    ## SPECIFY MODEL ##
                    'model_name': model_name,
                    'model': model,
                    'training_sample_size': training_sample_size,
                    'validation_sample_size': validation_sample_size,

                    ## SPECIFY TILE AND SCHEME ##
                    'tile': tile_coiid[0],
                    'years': [2018, 2019, 2020, 2021, 2022],
                    'scheme_name': scheme_name,
                    'crop_of_interest_id': tile_coiid[1], 
                    'in_season': in_season
                    }

                #fit_predict_report(**p) # run with the above parameters
                fit_predict_report(**p)
                
duration = default_timer() - start
print(duration)
with open(f'../data/times/time{start}.txt', 'a') as f:
    f.write(str(duration))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




