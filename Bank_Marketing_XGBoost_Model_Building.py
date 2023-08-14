#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


df = pd.read_csv(r"C:\Users\Aditya-pc\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional-full.csv",delimiter=';')


# In[11]:


df.head()


# In[21]:


corr_matrix = df.corr()


# In[22]:


corr_matrix


# In[23]:


df.columns


# In[24]:


df['y']


# In[28]:


cols_to_drop = ['duration','emp.var.rate','cons.price.idx','euribor3m','nr.employed']
df=df.drop(columns=cols_to_drop).rename(columns={'job':'job_type','default':'default_status','housing':'housing_loan_status','loan':'personal_loan_status','contact':'contact_type','month':'contact_month','day_of_week':'contact_day_of_week','campaign':'num_contacts','pdays':'days_last_contact','previous':'previous_contacts','poutcome':'previous_outcome','y':'result'})


# In[29]:


df['result']=df['result'].replace({'yes':1, 'no':0})


# In[30]:


df.head()


# In[31]:


df.info()


# In[32]:


df['result'].value_counts()


# In[33]:


from sklearn.model_selection import train_test_split

X=df.drop(columns='result')
y=df['result']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify = y, random_state=8)


# In[34]:


#building the pipeline for training


# In[37]:


#pip install category_encoders


# In[39]:


pip install xgboost


# In[40]:


from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier

estimators = [
    ('encoder', TargetEncoder()),
    ('clf', XGBClassifier(random_state=8)) #customizes objective func with the objective parameter
]
pipe = Pipeline(steps=estimators)
pipe


# In[41]:


Pipeline(steps=[('encoder', TargetEncoder()),
               ('clf',
               XGBClassifier(base_score=None, booster=None,
                            colsample_bylevel=None, colsample_bynode=None,
                            colsample_bytree=None, enable_categorical=False,
                            gamma= None, gpu_id = None, importance_type = None,
                            interaction_constraints=None, learning_rate=None,
                            max_delta_step=None, max_depth = None,
                            min_child_weight = None, missing = None,
                            monotone_constraints=None, n_estimators = 100,
                            n_jobs=None, num_parallel_tree = None,
                            predictor=None, random_state=8, reg_alpha=None,
                            reg_lambda = None, scale_pos_weight=None,
                            subsample=None, tree_method=None, validate_parameters=None,
                            verbosity=None))])


# In[42]:


### Setup hyperparameter tuning


# In[47]:


pip install scikit-optimize


# In[49]:


from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {
    'clf__max_depth' : Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior = 'log-uniform'),
    'clf__subsample' : Real(0.5,1.0),
    'clf__colsample_bytree': Real(0.5,1.0),
    'clf__colsample_bylevel': Real(0.5,1.0),
    'clf__colsampple_bynode': Real(0.5,1.0),
    'clf__reg_alpha': Real(0.0,10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8)


# In[50]:


###Train the XGBoost model


# In[51]:


opt.fit(X_train, y_train)


# In[52]:


### Evaluate the model and make predictions


# In[54]:


opt.best_estimator_


# In[55]:


opt.best_score_


# In[57]:


opt.score(X_test, y_test)


# In[58]:


opt.predict(X_test)


# In[59]:


opt.predict_proba(X_test)


# In[60]:


###Measure feature importance


# In[62]:


opt.best_estimator_.steps


# In[64]:


from xgboost import plot_importance

xgboost_step = opt.best_estimator_.steps[1]
xgboost_model = xgboost_step[1]
plot_importance(xgboost_model)


# In[ ]:




