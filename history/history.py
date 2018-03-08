list_result_hyperopt
loss
y
np.mean(y)
np.mean((y-np.mean(y))**2)
np.sqrt(np.mean((y-np.mean(y))**2))
y_train
y_pred
y_pred = xgb_hyperopt1.predict(X_train)
y_pred
mean_squared_error(y_train.values, y_pred)
mean_squared_error(y_train.values(), y_pred)
mean_squared_error(y_train.values, y_pred)
y_pred
y_train.values

np.mean((y_pred-y_train.values)**2)
np.sqrt(mean_squared_error(y_train.values, y_pred))
plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
%matplotlib inline
np.histogram
np.histogram(y_pred,y_train)
np.histogram(y_pred_y_train)
np.histogram(y_pred_y-train)
np.histogram(y_pred-y_train)
np.histogram(np.abs(y_pred-y_train),range(20))
np.histogram(np.abs(y_pred-y_train),range(100))
np.histogram(np.abs(y_pred-y_test),range(100))
history


Last login: Thu Mar  8 10:21:18 on ttys001
(python3) jordanvidal@MacBook-Pro-de-Jordan-2:~$ ipython
Python 3.6.2 (default, Jul 17 2017, 16:44:45)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.1.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: %paste
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command line best_time_to_call_run_fit script.

This script creates the command line which can take a date entry (optional).
It generates the model from all raw data before the given date.
"""

# Import from the standard library
import glob, re
import numpy as np
import pandas as pd
import xgboost
from datetime import datetime
from xgboost import XGBRegressor
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score
import logging
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename='report/output.log', level=logging.DEBUG, format='%(asctime)s :: %(levelname)s :: %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info('DATA IMPORT...')
data = {
    'tra': pd.read_csv('data/air_visit_data.csv.zip', compression="zip"),
    'as': pd.read_csv('data/air_store_info.csv.zip', compression="zip"),
    'hs': pd.read_csv('data/hpg_store_info.csv.zip', compression="zip"),
    'ar': pd.read_csv('data/air_reserve.csv.zip', compression="zip"),
    'hr': pd.read_csv('data/hpg_reserve.csv.zip', compression="zip"),
    'id': pd.read_csv('data/store_id_relation.csv.zip', compression="zip"),
    'tes': pd.read_csv('data/sample_submission.csv.zip', compression="zip"),
    'hol': pd.read_csv('data/date_info.csv.zip', compression="zip").rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

logging.info('DATA CLEANING + FE...')
for df in ['ar','hr']:
    #convert to datetime + creation de "reserve_datetime_diff"
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    # Ajout de la somme (tmp1) et de la moyenne (tmp2) du nombre de visiteurs et de reserve_datetime_diff par visites
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])

stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
#Label encoder pour les air_genre_name and air_area_name
lbl = preprocessing.LabelEncoder()
for i in range(10):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

#creation of train and test by merging train/test and stores dataset
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])
train = pd.merge(train, stores, how='left', on=['air_store_id','dow'])
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date'])
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

#Calcul de la somme, moyenne du nombre de visiteurs + moyenne du nombres de jours d'ecart entre reservation et visite au restaurant
train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

train['lon_plus_lat'] = train['longitude'] + train['latitude']
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

#END FEATURE ENGINEERING

## -- End pasted text --

---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-1-d19f2ec1cd31> in <module>()
     33 warnings.filterwarnings("ignore")
     34
---> 35 logging.basicConfig(filename='report/output.log', level=logging.DEBUG, format='%(asctime)s :: %(levelname)s :: %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
     36
     37 logging.info('DATA IMPORT...')

/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/logging/__init__.py in basicConfig(**kwargs)
   1779                 mode = kwargs.pop("filemode", 'a')
   1780                 if filename:
-> 1781                     h = FileHandler(filename, mode)
   1782                 else:
   1783                     stream = kwargs.pop("stream", None)

/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/logging/__init__.py in __init__(self, filename, mode, encoding, delay)
   1028             self.stream = None
   1029         else:
-> 1030             StreamHandler.__init__(self, self._open())
   1031
   1032     def close(self):

/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/logging/__init__.py in _open(self)
   1057         Return the resulting stream.
   1058         """
-> 1059         return open(self.baseFilename, self.mode, encoding=self.encoding)
   1060
   1061     def emit(self, record):

FileNotFoundError: [Errno 2] No such file or directory: '/Users/jordanvidal/report/output.log'

In [2]:

In [2]: pwd
Out[2]: '/Users/jordanvidal'

In [3]: cd Desktop/
/Users/jordanvidal/Desktop

In [4]: ls
DSSP7/                                  Screen Shot 2018-03-07 at 20.06.57.png
DataLab/                                Screen Shot 2018-03-07 at 20.10.59.png
DataScience/                            Screen Shot 2018-03-07 at 22.34.27.png
Mémoire de certification.pages          explo_ready.py
PERSO/                                  exploration_end.py
R/                                      fiche_logistic_reg.pdf
Screen Shot 2018-03-05 at 10.01.04.png  final-Copy1.ipynb
Screen Shot 2018-03-05 at 11.39.25.png  nicolas.py

In [5]: cd DataScience/
/Users/jordanvidal/Desktop/DataScience

In [6]: ls
Icon?   Jordan/

In [7]: cd Jordan/
/Users/jordanvidal/Desktop/DataScience/Jordan

In [8]: ls
code/ data/

In [9]: cd code/
/Users/jordanvidal/Desktop/DataScience/Jordan/code

In [10]: ls
Datalab/

In [11]: cd Datalab/
/Users/jordanvidal/Desktop/DataScience/Jordan/code/Datalab

In [12]: ls
Formation/                              date_time_test.ipynb*
FormationDataScientistJordan/           date_time_test2.ipynb*
Kaggle/                                 location_201705.csv*
R.ipynb*                                notebook_training/
Recruit Restaurant Visitor Forecasting/ notebook_training2/
Test_packages/                          pyPF/
Udemy_API/                              signals_201705.csv*
connected_car/                          titanic.ipynb*
connected_cars.ipynb

In [13]: cd Recruit\ Restaurant\ Visitor\ Forecasting
/Users/jordanvidal/Desktop/DataScience/Jordan/code/Datalab/Recruit Restaurant Visitor Forecasting

In [14]: ls
1.4 exploration.py        exploration20180703_2.py  img/                      submission.csv
Resumé.ipynb              exploration_nicolas.py    kaggle/                   viz.ipynb
data/                     final(1).ipynb            nohup.out                 viz_screen/
dssp7/                    final.ipynb               old/                      winner.py
exploration20180703.py    final_withviz.ipynb       report/

In [15]: %paste
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command line best_time_to_call_run_fit script.

This script creates the command line which can take a date entry (optional).
It generates the model from all raw data before the given date.
"""

# Import from the standard library
import glob, re
import numpy as np
import pandas as pd
import xgboost
from datetime import datetime
from xgboost import XGBRegressor
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score
import logging
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename='report/output.log', level=logging.DEBUG, format='%(asctime)s :: %(levelname)s :: %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info('DATA IMPORT...')
data = {
    'tra': pd.read_csv('data/air_visit_data.csv.zip', compression="zip"),
    'as': pd.read_csv('data/air_store_info.csv.zip', compression="zip"),
    'hs': pd.read_csv('data/hpg_store_info.csv.zip', compression="zip"),
    'ar': pd.read_csv('data/air_reserve.csv.zip', compression="zip"),
    'hr': pd.read_csv('data/hpg_reserve.csv.zip', compression="zip"),
    'id': pd.read_csv('data/store_id_relation.csv.zip', compression="zip"),
    'tes': pd.read_csv('data/sample_submission.csv.zip', compression="zip"),
    'hol': pd.read_csv('data/date_info.csv.zip', compression="zip").rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

logging.info('DATA CLEANING + FE...')
for df in ['ar','hr']:
    #convert to datetime + creation de "reserve_datetime_diff"
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    # Ajout de la somme (tmp1) et de la moyenne (tmp2) du nombre de visiteurs et de reserve_datetime_diff par visites
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])

stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
#Label encoder pour les air_genre_name and air_area_name
lbl = preprocessing.LabelEncoder()
for i in range(10):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

#creation of train and test by merging train/test and stores dataset
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])
train = pd.merge(train, stores, how='left', on=['air_store_id','dow'])
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date'])
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

#Calcul de la somme, moyenne du nombre de visiteurs + moyenne du nombres de jours d'ecart entre reservation et visite au restaurant
train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

train['lon_plus_lat'] = train['longitude'] + train['latitude']
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

#END FEATURE ENGINEERING

## -- End pasted text --


In [16]:

In [16]: %paste
y = train.pop('visitors')
X_train, X_test, y_train, y_test = train_test_split(train[col], y, test_size=0.2, random_state=42)

## -- End pasted text --

In [17]: %paste
def test_model(model, X_test, y_test):
    p_test = model.predict_proba(X_test)
    return mean_squared_error(y_test, p_test.argmax(axis=1))

## -- End pasted text --

In [18]: %paste
rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, min_samples_leaf=10)
logging.info('2. RandomForestRegressor- start fit')
rf.fit(X_train, y_train)

## -- End pasted text --
Out[18]:
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=10, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

In [19]: %paste
score_rf = test_model(rf, X_test, y_test)

## -- End pasted text --
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-19-284493e03896> in <module>()
----> 1 score_rf = test_model(rf, X_test, y_test)

<ipython-input-17-a233a2d45f3b> in test_model(model, X_test, y_test)
      1 def test_model(model, X_test, y_test):
----> 2     p_test = model.predict_proba(X_test)
      3     return mean_squared_error(y_test, p_test.argmax(axis=1))

AttributeError: 'RandomForestRegressor' object has no attribute 'predict_proba'

In [20]: print(rf.predict(y_test))
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-20-ac1559633911> in <module>()
----> 1 print(rf.predict(y_test))

~/python3/lib/python3.6/site-packages/sklearn/ensemble/forest.py in predict(self, X)
    679         check_is_fitted(self, 'estimators_')
    680         # Check data
--> 681         X = self._validate_X_predict(X)
    682
    683         # Assign chunk of trees to jobs

~/python3/lib/python3.6/site-packages/sklearn/ensemble/forest.py in _validate_X_predict(self, X)
    355                                  "call `fit` before exploiting the model.")
    356
--> 357         return self.estimators_[0]._validate_X_predict(X, check_input=True)
    358
    359     @property

~/python3/lib/python3.6/site-packages/sklearn/tree/tree.py in _validate_X_predict(self, X, check_input)
    371         """Validate X whenever one tries to predict, apply, predict_proba"""
    372         if check_input:
--> 373             X = check_array(X, dtype=DTYPE, accept_sparse="csr")
    374             if issparse(X) and (X.indices.dtype != np.intc or
    375                                 X.indptr.dtype != np.intc):

~/python3/lib/python3.6/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
    439                     "Reshape your data either using array.reshape(-1, 1) if "
    440                     "your data has a single feature or array.reshape(1, -1) "
--> 441                     "if it contains a single sample.".format(array))
    442             array = np.atleast_2d(array)
    443             # To ensure that array flags are maintained

ValueError: Expected 2D array, got 1D array instead:
array=[ 33.  20.  24. ...,  24.  18.   1.].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

In [21]: %paste
rf.predict(X_test)

## -- End pasted text --
Out[21]:
array([ 21.75353785,  14.76308219,  23.05401772, ...,  24.35914754,
        15.42929567,  14.13926417])

In [22]: %paste
train_error_rf = round(mean_squared_error(y_train, rf.predict(X_train)), 3)
test_error_rf = round(mean_squared_error(y_test, rf.predict(X_test)), 3)

## -- End pasted text --

In [23]: train_error_rf
Out[23]: 82.879000000000005

In [24]: test_error_rf
Out[24]: 118.792

In [25]: %paste
score_rf = test_model(rf, X_test, y_test)
## -- End pasted text --
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-25-284493e03896> in <module>()
----> 1 score_rf = test_model(rf, X_test, y_test)

<ipython-input-17-a233a2d45f3b> in test_model(model, X_test, y_test)
      1 def test_model(model, X_test, y_test):
----> 2     p_test = model.predict_proba(X_test)
      3     return mean_squared_error(y_test, p_test.argmax(axis=1))

AttributeError: 'RandomForestRegressor' object has no attribute 'predict_proba'

In [26]: %paste
rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, min_samples_leaf=10)

## -- End pasted text --
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-26-c9ed1a8d8c02> in <module>()
----> 1 rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, min_samples_leaf=10)

NameError: name 'RandomForestClassifier' is not defined

In [27]: from sklearn.ensemble import RandomForestClassifier

In [28]: rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, min_samples_leaf=10)

In [29]: rf.fit(X_train, y_train)
Out[29]:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=10, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

In [30]: %paste
score_rf = test_model(rf, X_test, y_test)
print('Random Forest score: {}'.format(score_rf))

## -- End pasted text --
Random Forest score: 163.00285589623576

In [31]: %paste
rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, min_samples_leaf=10)
## -- End pasted text --

In [32]: %paste
rf.fit(X_train, y_train)
## -- End pasted text --
Out[32]:
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=10, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

In [33]: %paste
score_rf = test_model(rf, X_test, y_test)
## -- End pasted text --
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-33-284493e03896> in <module>()
----> 1 score_rf = test_model(rf, X_test, y_test)

<ipython-input-17-a233a2d45f3b> in test_model(model, X_test, y_test)
      1 def test_model(model, X_test, y_test):
----> 2     p_test = model.predict_proba(X_test)
      3     return mean_squared_error(y_test, p_test.argmax(axis=1))

AttributeError: 'RandomForestRegressor' object has no attribute 'predict_proba'

In [34]: %paste
rf.predict(X_test)

## -- End pasted text --
Out[34]:
array([ 23.2318163 ,  13.07517554,  22.21015306, ...,  26.35879395,
        16.78870175,  12.63530777])

In [35]: %paste
train_error_rf = round(mean_squared_error(y_train, rf.predict(X_train)), 3)
test_error_rf = round(mean_squared_error(y_test, rf.predict(X_test)), 3)

## -- End pasted text --

In [36]: train_error_rf
Out[36]: 82.795000000000002

In [37]: test_error_rf
Out[37]: 118.051

In [38]: print(rf.feature_importances_)
[  5.49515140e-03   2.19139455e-04   1.05359951e-02   3.83116411e-03
   4.49128102e-03   7.60024133e-03   7.86298888e-01   1.41386231e-02
   1.74887597e-02   7.32189230e-03   3.18990400e-03   4.34242654e-03
   3.15079915e-03   3.06310507e-03   3.05889335e-03   5.84505527e-04
   4.07432656e-03   5.09879830e-04   1.28439364e-04   4.93906945e-03
   0.00000000e+00   1.18326265e-03   0.00000000e+00   6.60598179e-03
   0.00000000e+00   5.54400544e-04   0.00000000e+00   7.94810942e-05
   0.00000000e+00   0.00000000e+00   8.12448175e-04   0.00000000e+00
   7.67769653e-04   0.00000000e+00   1.05754208e-03   2.72013283e-02
   1.39914078e-03   1.65933425e-03   9.95256112e-04   1.27568748e-03
   7.83928726e-04   1.62739924e-03   1.81423245e-04   1.45008667e-04
   7.57453792e-05   4.79101889e-02   2.24976403e-03   3.10544115e-03
   3.15568204e-03   1.27113011e-02]

In [39]: print(rf.predict(X_test))
[ 23.2318163   13.07517554  22.21015306 ...,  26.35879395  16.78870175
  12.63530777]

In [40]: %paste
print('RandomForest feature importance: {}'.format(rf.feature_importances_))
logging.info('2.1 Feature importances...')
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
dtrain = xgboost.DMatrix(train[col], label=y)

## -- End pasted text --
RandomForest feature importance: [  5.49515140e-03   2.19139455e-04   1.05359951e-02   3.83116411e-03
   4.49128102e-03   7.60024133e-03   7.86298888e-01   1.41386231e-02
   1.74887597e-02   7.32189230e-03   3.18990400e-03   4.34242654e-03
   3.15079915e-03   3.06310507e-03   3.05889335e-03   5.84505527e-04
   4.07432656e-03   5.09879830e-04   1.28439364e-04   4.93906945e-03
   0.00000000e+00   1.18326265e-03   0.00000000e+00   6.60598179e-03
   0.00000000e+00   5.54400544e-04   0.00000000e+00   7.94810942e-05
   0.00000000e+00   0.00000000e+00   8.12448175e-04   0.00000000e+00
   7.67769653e-04   0.00000000e+00   1.05754208e-03   2.72013283e-02
   1.39914078e-03   1.65933425e-03   9.95256112e-04   1.27568748e-03
   7.83928726e-04   1.62739924e-03   1.81423245e-04   1.45008667e-04
   7.57453792e-05   4.79101889e-02   2.24976403e-03   3.10544115e-03
   3.15568204e-03   1.27113011e-02]

In [41]: %paste
params_clf = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params_clf)
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

## -- End pasted text --
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-41-cfc34ae2cfcc> in <module>()
      1 params_clf = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
      2           'learning_rate': 0.01, 'loss': 'ls'}
----> 3 clf = ensemble.GradientBoostingRegressor(**params_clf)
      4 clf.fit(X_train, y_train)
      5 mse = mean_squared_error(y_test, clf.predict(X_test))

NameError: name 'ensemble' is not defined

In [42]: %paste
from sklearn import preprocessing, ensemble
## -- End pasted text --

In [43]: %paste
params_clf = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params_clf)
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

## -- End pasted text --

MSE: 122.6056

In [44]:

In [44]: %paste
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
mse_xgb = mean_squared_error(y_test, xgb.predict(X_test))
print('XGB score: {}'.format(mse_xgb))
train_error_xgb = round(mean_squared_error(y_train, xgb.predict(X_train)), 3)
test_error_xgb = round(mean_squared_error(y_test, xgb.predict(X_test)), 3)
print('XGBoost without Grid Search train error: {}'.format(train_error_xgb))
print('XGBoost without Grid Search test error: {}'.format(test_error_xgb))

## -- End pasted text --

XGB score: 122.21274670518012
XGBoost without Grid Search train error: 111.321
XGBoost without Grid Search test error: 122.213

In [45]:

In [45]: %paste
params = grid_search.best_params_
xgb_grid = XGBRegressor(**params)
xgb_grid.fit(X_train, y_train)
train_error_xgb_grid = round(median_absolute_error(y_train, xgb_grid.predict(X_train)), 3)
test_error_xgb_grid = round(median_absolute_error(y_test, xgb_grid.predict(X_test)), 3)
print("XGBoost with Grid Search train error: {}".format(train_error_xgb_grid))
print("XGBoost with Grid Search test error: {}".format(test_error_xgb_grid))

## -- End pasted text --
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-45-f5683e399791> in <module>()
----> 1 params = grid_search.best_params_
      2 xgb_grid = XGBRegressor(**params)
      3 xgb_grid.fit(X_train, y_train)
      4 train_error_xgb_grid = round(median_absolute_error(y_train, xgb_grid.predict(X_train)), 3)
      5 test_error_xgb_grid = round(median_absolute_error(y_test, xgb_grid.predict(X_test)), 3)

NameError: name 'grid_search' is not defined

In [46]: %paste
params = dict(max_depth=list(range(5, 10)), n_estimators=[100], learning_rate=[0.1])
logging.info('4. Start XGBoost Grid Search')
grid_search = GridSearchCV(xgb, param_grid=params, n_jobs=4).fit(X_train, y_train)
logging.info('4. End XGBoost Grid Search')
# summarize the results of the grid search
print('Best Estimator: {}'.format(grid_search.best_estimator_))
print('Best Parameters: {}'.format(grid_search.best_params_))
print('Best Score: {}'.format(grid_search.best_score_))
print(sorted(grid_search.cv_results_.keys()))

## -- End pasted text --


Best Estimator: XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=8, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
Best Parameters: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 100}
Best Score: 0.59894293112934
['mean_fit_time', 'mean_score_time', 'mean_test_score', 'mean_train_score', 'param_learning_rate', 'param_max_depth', 'param_n_estimators', 'params', 'rank_test_score', 'split0_test_score', 'split0_train_score', 'split1_test_score', 'split1_train_score', 'split2_test_score', 'split2_train_score', 'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score']

In [47]:

In [47]: test_error_xgb
Out[47]: 122.21299999999999

In [48]: %paste
params = grid_search.best_params_
xgb_grid = XGBRegressor(**params)
xgb_grid.fit(X_train, y_train)
train_error_xgb_grid = round(median_absolute_error(y_train, xgb_grid.predict(X_train)), 3)
test_error_xgb_grid = round(median_absolute_error(y_test, xgb_grid.predict(X_test)), 3)
print("XGBoost with Grid Search train error: {}".format(train_error_xgb_grid))
print("XGBoost with Grid Search test error: {}".format(test_error_xgb_grid))

## -- End pasted text --

XGBoost with Grid Search train error: 4.486
XGBoost with Grid Search test error: 4.629

In [49]:

In [49]: perf = round((test_error_xgb/test_error_xgb_grid)*100, 2)

In [50]: perf
Out[50]: 2640.1599999999999

In [51]: perf = round((test_error_xgb/test_error_xgb_grid), 2)

In [52]: perf
Out[52]: 26.399999999999999

In [53]: %paste
params = dict(max_depth=list(range(8, 12)),
              n_estimators=list(range(30, 110, 10)))
mygcv = GridSearchCV(rf, param_grid=params).fit(X_train, y_train)

## -- End pasted text --


In [54]:

In [54]: %paste
print('Tested parameters: {}'.format(mygcv.cv_results_['params']))
print('Best Estimator: {}'.format(mygcv.best_estimator_))
print('Best Parameters: {}'.format(mygcv.best_params_))

## -- End pasted text --
Tested parameters: [{'max_depth': 8, 'n_estimators': 30}, {'max_depth': 8, 'n_estimators': 40}, {'max_depth': 8, 'n_estimators': 50}, {'max_depth': 8, 'n_estimators': 60}, {'max_depth': 8, 'n_estimators': 70}, {'max_depth': 8, 'n_estimators': 80}, {'max_depth': 8, 'n_estimators': 90}, {'max_depth': 8, 'n_estimators': 100}, {'max_depth': 9, 'n_estimators': 30}, {'max_depth': 9, 'n_estimators': 40}, {'max_depth': 9, 'n_estimators': 50}, {'max_depth': 9, 'n_estimators': 60}, {'max_depth': 9, 'n_estimators': 70}, {'max_depth': 9, 'n_estimators': 80}, {'max_depth': 9, 'n_estimators': 90}, {'max_depth': 9, 'n_estimators': 100}, {'max_depth': 10, 'n_estimators': 30}, {'max_depth': 10, 'n_estimators': 40}, {'max_depth': 10, 'n_estimators': 50}, {'max_depth': 10, 'n_estimators': 60}, {'max_depth': 10, 'n_estimators': 70}, {'max_depth': 10, 'n_estimators': 80}, {'max_depth': 10, 'n_estimators': 90}, {'max_depth': 10, 'n_estimators': 100}, {'max_depth': 11, 'n_estimators': 30}, {'max_depth': 11, 'n_estimators': 40}, {'max_depth': 11, 'n_estimators': 50}, {'max_depth': 11, 'n_estimators': 60}, {'max_depth': 11, 'n_estimators': 70}, {'max_depth': 11, 'n_estimators': 80}, {'max_depth': 11, 'n_estimators': 90}, {'max_depth': 11, 'n_estimators': 100}]
Best Estimator: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=11,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=10, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=90, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
Best Parameters: {'max_depth': 11, 'n_estimators': 90}

In [55]: %paste
rf2 = RandomForestRegressor(n_estimators=90, max_depth=11, random_state=42)
rf2.fit(X_train, y_train)
train_error_rf2 = round(mean_squared_error(y_train, rf2.predict(X_train)), 3)
test_error_rf2 = round(mean_squared_error(y_test, rf2.predict(X_test)), 3)
print("train error: {}".format(train_error_rf2))
print("test error with Best Parameters: {}".format(test_error_rf2))
print("test error without Best Parameters: {}".format(test_error_rf))

## -- End pasted text --
train error: 94.33
test error with Best Parameters: 121.52
test error without Best Parameters: 118.051

In [56]: rf2 = RandomForestRegressor(params=mygcv.best_params_)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-56-d2df072557cc> in <module>()
----> 1 rf2 = RandomForestRegressor(params=mygcv.best_params_)

TypeError: __init__() got an unexpected keyword argument 'params'

In [57]: rf2 = RandomForestRegressor(params_clf=mygcv.best_params_)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-57-f8be260ae7d5> in <module>()
----> 1 rf2 = RandomForestRegressor(params_clf=mygcv.best_params_)

TypeError: __init__() got an unexpected keyword argument 'params_clf'

In [58]: mygcv.best_params_
Out[58]: {'max_depth': 11, 'n_estimators': 90}

In [59]: %paste
def get_final_parameters(best_params, origin_model_d, current_model):
    """
    Fix the string parameters returned by hyperopt.
    Use the index given by hyperopt to find the real string value
    for a specific parameter
    """
    for element_p in list(best_params.keys()):
        if isinstance(
                origin_model_d[current_model][1][element_p],
                hyperopt.pyll.base.Apply):
            if origin_model_d[current_model][1][element_p].name == "float":
                pass
            elif origin_model_d[current_model][1][element_p].name == "switch":
                apply_obj = origin_model_d[current_model][1][element_p]
                literal_obj = apply_obj.pos_args[best_params[element_p] + 1]
                best_params[element_p] = literal_obj.obj
            else:
                pass
        else:
            pass
    return best_params

def regression_params_opt(
        origin_model_d,
        current_model,
        X_train,
        X_test,
        y_train,
        y_test):
    print(current_model)
    best = 0
    global i
    i = 0
    max_eval = 10
    trials = Trials()

    def rmse_score(params):
        global i
        model_fit = origin_model_d[current_model][0](
            **params).fit(x_train, y_train,eval_set=[(x_train,y_train),(x_test,y_test),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(x_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        params['best_iteration'] = model_fit.best_iteration
        df_result_hyperopt.loc[i] = np.append(loss, list(params.values()))
        i = i + 1
        return {'loss': loss, 'status': STATUS_OK}

    def rmse_score_nicolas(params):
        global i
        print("origin_model_d[current_model]",origin_model_d[current_model][0](**params))
        v = np.array(range(x_train.shape[0]))%2==0
        #v = [e%2==0 for e in range(x_train.shape[0]]
        model_fit = origin_model_d[current_model][0](
            **params).fit(x_train[v], y_train[v],eval_set=[(x_train[v],y_train[v]),(x_train[v==False],y_train[v==False]),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(x_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        df_result_hyperopt.loc[i] = np.append(loss, list(params.values()))
        i = i + 1
        return {'loss': loss, 'status': STATUS_OK}

    df_result_hyperopt = pd.DataFrame(
        columns=[
            np.append(
                'score', list(
                    origin_model_d[current_model][1].keys()))])

    best = fmin(rmse_score,
                origin_model_d[current_model][1],
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials)
    print(best)
    print(best.keys())
    print("\n")
    return get_final_parameters(best, origin_model_d, current_model)

if __name__ == '__main__':

    xgbr_d = {'gamma': hp.quniform('gamma', 0.0, 5.0, 0.1),
              'learning_rate': hp.choice('learning_rate', [0.1]),
              'colsample_bytree': hp.quniform('colsample_bytree',
                                              0.3,
                                              1.,
                                              0.05),
              'max_depth': hp.choice('max_depth', list(range(5, 10))),
              'min_child_weight': hp.quniform('min_child_weight', 1., 5., 1),
              'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
              'nthread': hp.choice('nthread', [-1]),
              'n_estimators': hp.choice('n_estimators', [1500]),
              'objective': hp.choice('objective', ['reg:linear']),
              'reg_lambda': hp.quniform('reg_alpha', 0.0, 4.0, 0.1),
              'reg_alpha': hp.quniform('reg_lambda', 0.0, 4.0, 0.1)}

    base_model = {"XGBRegressor": [XGBRegressor, xgbr_d]}


    #x_train = train[col]
    #y_train = train["visitors"]

    #x_test = test[col]
    #y_test = test["visitors"]

    hyper_parametres = regression_params_opt(
        origin_model_d=base_model,
        current_model="XGBRegressor",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

aaa = df_result_hyperopt[:]

df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")
## -- End pasted text --
XGBRegressor
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-59-1dc4aad51f92> in <module>()
    107         X_test=X_test,
    108         y_train=y_train,
--> 109         y_test=y_test
    110     )
    111

<ipython-input-59-1dc4aad51f92> in regression_params_opt(origin_model_d, current_model, X_train, X_test, y_train, y_test)
     69                 algo=tpe.suggest,
     70                 max_evals=max_eval,
---> 71                 trials=trials)
     72     print(best)
     73     print(best.keys())

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    305             verbose=verbose,
    306             catch_eval_exceptions=catch_eval_exceptions,
--> 307             return_argmin=return_argmin,
    308         )
    309

~/python3/lib/python3.6/site-packages/hyperopt/base.py in fmin(self, fn, space, algo, max_evals, rstate, verbose, pass_expr_memo_ctrl, catch_eval_exceptions, return_argmin)
    633             pass_expr_memo_ctrl=pass_expr_memo_ctrl,
    634             catch_eval_exceptions=catch_eval_exceptions,
--> 635             return_argmin=return_argmin)
    636
    637

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    318                     verbose=verbose)
    319     rval.catch_eval_exceptions = catch_eval_exceptions
--> 320     rval.exhaust()
    321     if return_argmin:
    322         return trials.argmin

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in exhaust(self)
    197     def exhaust(self):
    198         n_done = len(self.trials)
--> 199         self.run(self.max_evals - n_done, block_until_done=self.async)
    200         self.trials.refresh()
    201         return self

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in run(self, N, block_until_done)
    171             else:
    172                 # -- loop over trials and do the jobs directly
--> 173                 self.serial_evaluate()
    174
    175             if stopped:

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in serial_evaluate(self, N)
     90                 ctrl = base.Ctrl(self.trials, current_trial=trial)
     91                 try:
---> 92                     result = self.domain.evaluate(spec, ctrl)
     93                 except Exception as e:
     94                     logger.info('job exception: %s' % str(e))

~/python3/lib/python3.6/site-packages/hyperopt/base.py in evaluate(self, config, ctrl, attach_attachments)
    838                 memo=memo,
    839                 print_node_on_error=self.rec_eval_print_node_on_error)
--> 840             rval = self.fn(pyll_rval)
    841
    842         if isinstance(rval, (float, int, np.number)):

<ipython-input-59-1dc4aad51f92> in rmse_score(params)
     38         global i
     39         model_fit = origin_model_d[current_model][0](
---> 40             **params).fit(x_train, y_train,eval_set=[(x_train,y_train),(x_test,y_test),],early_stopping_rounds=50)
     41         y_pred_train = model_fit.predict(x_test)
     42         loss = mean_squared_error(y_test, y_pred_train)**0.5

NameError: name 'x_train' is not defined

In [60]: X_train
Out[60]:
        dow  year  month  day_of_week  holiday_flg  min_visitors  \
107651    2  2017      3            6            0           9.0
163691    5  2016     10            2            0          13.0
92590     1  2016     11            5            0           3.0
226442    3  2016      5            4            0           1.0
28716     6  2016      9            3            0           2.0
139342    5  2016      5            2            0           3.0
24184     2  2016      9            6            0           3.0
78564     4  2017      3            0            0           6.0
34156     0  2016      5            1            0           2.0
186975    3  2017      3            4            0           1.0
228618    1  2017      3            5            0           4.0
51795     5  2016     10            2            0           1.0
200350    1  2016     11            5            0          15.0
55685     5  2016      8            2            0           1.0
81228     5  2017      1            2            0           1.0
112227    3  2016      9            4            0           3.0
228229    0  2016      2            1            0           5.0
2010      5  2017      1            2            0           4.0
32564     1  2016     10            5            0           2.0
165237    1  2016     10            5            0           1.0
124473    1  2016     10            5            0           1.0
186348    1  2016      9            5            0           1.0
111985    4  2017      3            0            0           1.0
47559     2  2016     10            6            0           2.0
190380    5  2016      2            2            0           1.0
198074    5  2017      4            2            0           2.0
171841    3  2016      9            4            0           4.0
180422    6  2016     12            3            0           8.0
61667     4  2017      2            0            0           4.0
103188    2  2016      3            6            0           1.0
...     ...   ...    ...          ...          ...           ...
184779    5  2016      7            2            0           1.0
214176    1  2017      1            5            0           1.0
235796    4  2016      2            0            0           4.0
103355    4  2016      9            0            0           1.0
5311      4  2016      7            0            0           2.0
199041    5  2016     11            2            0          16.0
64925     6  2016      6            3            0           1.0
194027    5  2016      8            2            0           4.0
59735     4  2016      2            0            0           2.0
769       6  2016      8            3            0          22.0
64820     2  2016     12            6            0           2.0
67221     3  2016      5            4            0           1.0
41090     3  2016      9            4            0          19.0
16023     5  2016      9            2            0           1.0
191335    0  2016     12            1            0           5.0
175203    3  2017      4            4            0          14.0
126324    1  2016      9            5            0           4.0
112727    6  2016     11            3            0           2.0
87498     5  2017      2            2            0           2.0
168266    3  2017      2            4            0           2.0
213458    5  2017      1            2            0           8.0
137337    2  2016      7            6            0           1.0
54886     0  2016      6            1            0           1.0
207892    3  2017      4            4            0           2.0
110268    4  2016      4            0            0           5.0
119879    5  2016     12            2            0           2.0
103694    4  2017      2            0            0           3.0
131932    2  2017      3            6            0           1.0
146867    3  2016      9            4            0           1.0
121958    0  2017      1            1            1           1.0

        mean_visitors  median_visitors  max_visitors  count_observations  \
107651      23.268293             23.0          38.0                41.0
163691      48.255814             51.0          82.0                43.0
92590       10.216667             10.0          25.0                60.0
226442      11.529412              9.0          49.0                68.0
28716       31.625000             30.5          52.0                40.0
139342      16.029412             13.5          39.0                68.0
24184       15.850746             15.0          42.0                67.0
78564       17.000000             17.5          30.0                42.0
34156       26.030303             26.0          70.0                66.0
186975       7.046154              6.0          19.0                65.0
228618      23.815385             23.0          48.0                65.0
51795        5.112903              5.0          14.0                62.0
200350      22.585366             22.0          32.0                41.0
55685        6.641026              6.0          20.0                39.0
81228       25.000000             24.0          43.0                41.0
112227      14.791045             14.0          35.0                67.0
228229      23.323077             23.0          46.0                65.0
2010        21.350877             16.0          86.0                57.0
32564       14.975000             15.0          43.0                40.0
165237       8.315789              8.5          23.0                38.0
124473      18.714286             17.0          47.0                42.0
186348       3.750000              3.0          16.0                40.0
111985       8.875000              6.0          46.0                56.0
47559       10.800000              9.5          26.0                40.0
190380      10.294118             10.0          24.0                51.0
198074      13.209302             13.0          26.0                43.0
171841      20.512195             20.0          54.0                41.0
180422      26.117647             25.5          61.0                68.0
61667       24.584615             25.0          42.0                65.0
103188      13.322034             11.0          45.0                59.0
...               ...              ...           ...                 ...
184779      13.135135             12.0          41.0                37.0
214176       8.107143              8.0          20.0                28.0
235796      15.208955             15.0          33.0                67.0
103355      11.593220              9.0          92.0                59.0
5311        19.950000             20.5          37.0                40.0
199041      42.790698             43.0          66.0                43.0
64925        9.200000              9.0          24.0                45.0
194027      10.813953             10.0          19.0                43.0
59735        8.880597              8.0          24.0                67.0
769         38.666667             37.0          66.0                42.0
64820       10.250000              4.5          31.0                12.0
67221        6.953125              7.0          21.0                64.0
41090       36.560976             37.0          55.0                41.0
16023       14.625000             15.0          30.0                64.0
191335      20.400000             20.0          40.0                55.0
175203      32.756098             34.0          68.0                41.0
126324      12.227273             12.5          21.0                66.0
112727      18.968750             15.0          81.0                64.0
87498        7.977273              6.0          24.0                44.0
168266       9.219512              8.0          29.0                41.0
213458      19.000000             19.0          39.0                42.0
137337      12.230769             13.0          27.0                39.0
54886        3.666667              2.5          11.0                 6.0
207892      12.119048             12.0          26.0                42.0
110268      48.223881             49.0          87.0                67.0
119879      11.853659             12.0          25.0                41.0
103694      16.023256             18.0          25.0                43.0
131932       8.225000              7.0          29.0                40.0
146867      29.897436             30.0          51.0                39.0
121958       5.877193              4.0          32.0                57.0

            ...        rs2_y  rv2_y  total_reserv_sum  total_reserv_mean  \
107651      ...         -1.0   -1.0              -1.0          -1.000000
163691      ...         -1.0   -1.0              -1.0          -1.000000
92590       ...         -1.0   -1.0              -1.0          -1.000000
226442      ...         -1.0   -1.0              -1.0          -1.000000
28716       ...         -1.0   -1.0              -1.0          -1.000000
139342      ...         -1.0   -1.0              -1.0          -1.000000
24184       ...         -1.0   -1.0              -1.0          -1.000000
78564       ...         -1.0   -1.0              -1.0          -1.000000
34156       ...         -1.0   -1.0              -1.0          -1.000000
186975      ...         -1.0   -1.0              -1.0          -1.000000
228618      ...         -1.0   -1.0              -1.0          -1.000000
51795       ...         -1.0   -1.0              -1.0          -1.000000
200350      ...         -1.0   -1.0              -1.0          -1.000000
55685       ...         -1.0   -1.0              -1.0          -1.000000
81228       ...         -1.0   -1.0              -1.0          -1.000000
112227      ...         -1.0   -1.0              -1.0          -1.000000
228229      ...         -1.0   -1.0              -1.0          -1.000000
2010        ...         -1.0   -1.0              -1.0          -1.000000
32564       ...         -1.0   -1.0              -1.0          -1.000000
165237      ...         -1.0   -1.0              -1.0          -1.000000
124473      ...         -1.0   -1.0              -1.0          -1.000000
186348      ...         -1.0   -1.0              -1.0          -1.000000
111985      ...         17.0   19.0              78.0          16.166667
47559       ...         -1.0   -1.0              -1.0          -1.000000
190380      ...         -1.0   -1.0              -1.0          -1.000000
198074      ...         -1.0   -1.0              -1.0          -1.000000
171841      ...         -1.0   -1.0              -1.0          -1.000000
180422      ...         -1.0   -1.0              -1.0          -1.000000
61667       ...         -1.0   -1.0              -1.0          -1.000000
103188      ...         -1.0   -1.0              -1.0          -1.000000
...         ...          ...    ...               ...                ...
184779      ...         -1.0   -1.0              -1.0          -1.000000
214176      ...         -1.0   -1.0              -1.0          -1.000000
235796      ...         -1.0   -1.0              -1.0          -1.000000
103355      ...         -1.0   -1.0              -1.0          -1.000000
5311        ...         -1.0   -1.0              -1.0          -1.000000
199041      ...         -1.0   -1.0              -1.0          -1.000000
64925       ...         -1.0   -1.0              -1.0          -1.000000
194027      ...         -1.0   -1.0              -1.0          -1.000000
59735       ...         -1.0   -1.0              -1.0          -1.000000
769         ...         -1.0   -1.0              -1.0          -1.000000
64820       ...         -1.0   -1.0              -1.0          -1.000000
67221       ...         -1.0   -1.0              -1.0          -1.000000
41090       ...         -1.0   -1.0              -1.0          -1.000000
16023       ...         -1.0   -1.0              -1.0          -1.000000
191335      ...         -1.0   -1.0              -1.0          -1.000000
175203      ...          1.0    1.0              -1.0          -1.000000
126324      ...         -1.0   -1.0              -1.0          -1.000000
112727      ...         -1.0   -1.0              -1.0          -1.000000
87498       ...         -1.0   -1.0              -1.0          -1.000000
168266      ...         -1.0   -1.0              -1.0          -1.000000
213458      ...         -1.0   -1.0              -1.0          -1.000000
137337      ...         -1.0   -1.0              -1.0          -1.000000
54886       ...         -1.0   -1.0              -1.0          -1.000000
207892      ...         -1.0   -1.0              -1.0          -1.000000
110268      ...         -1.0   -1.0              -1.0          -1.000000
119879      ...         -1.0   -1.0              -1.0          -1.000000
103694      ...         -1.0   -1.0              -1.0          -1.000000
131932      ...         -1.0   -1.0              -1.0          -1.000000
146867      ...         -1.0   -1.0              -1.0          -1.000000
121958      ...         -1.0   -1.0              -1.0          -1.000000

        total_reserv_dt_diff_mean  date_int  var_max_lat  var_max_long  \
107651                  -1.000000  20170308     8.326792      4.569849
163691                  -1.000000  20161008     9.263682      9.432221
92590                   -1.000000  20161108     9.309736      6.547459
226442                  -1.000000  20160519     8.326792      4.569849
28716                   -1.000000  20160918     9.634387     11.818381
139342                  -1.000000  20160521     9.341302      8.672386
24184                   -1.000000  20160928     9.263682      9.432221
78564                   -1.000000  20170317     8.358855      4.569348
34156                   -1.000000  20160516     9.634387     11.818381
186975                  -1.000000  20170330    10.431416     13.880585
228618                  -1.000000  20170321     8.321066      4.636960
51795                   -1.000000  20161022     9.254539      8.645298
200350                  -1.000000  20161122     9.534733     10.911086
55685                   -1.000000  20160827     8.358855      4.569348
81228                   -1.000000  20170128     8.374060      4.620151
112227                  -1.000000  20160915     6.104264      5.237072
228229                  -1.000000  20160222     8.321066      4.636960
2010                    -1.000000  20170107     8.362564      4.521799
32564                   -1.000000  20161025     8.277057      4.426219
165237                  -1.000000  20161025    10.808664     13.715490
124473                  -1.000000  20161004     9.205483      9.588045
186348                  -1.000000  20160913    10.431416     13.880585
111985                  14.333333  20170317     0.249996      1.908579
47559                   -1.000000  20161019     9.341302      8.672386
190380                  -1.000000  20160213     8.321066      4.636960
198074                  -1.000000  20170422     8.374060      4.620151
171841                  -1.000000  20160901     8.306618      4.865555
180422                  -1.000000  20161211     5.751556      3.402995
61667                   -1.000000  20170203     8.308025      4.493403
103188                  -1.000000  20160302     9.315270      8.763373
...                           ...       ...          ...           ...
184779                  -1.000000  20160702    10.463312     14.077843
214176                  -1.000000  20170131     8.411424      4.604037
235796                  -1.000000  20160212     9.339371      8.763597
103355                  -1.000000  20160930     9.315270      8.763373
5311                    -1.000000  20160722     8.362564      4.521799
199041                  -1.000000  20161119     8.294514      4.556793
64925                   -1.000000  20160612     8.308025      4.493403
194027                  -1.000000  20160827    10.431416     13.880585
59735                   -1.000000  20160205    10.431416     13.880585
769                     -1.000000  20160828     9.315270      8.763373
64820                   -1.000000  20161221     9.534733     10.911086
67221                   -1.000000  20160519     8.362564      4.521799
41090                   -1.000000  20160901     8.374060      4.620151
16023                   -1.000000  20160910     8.379169      4.575227
191335                  -1.000000  20161205     8.358855      4.569348
175203                  -1.000000  20170420     8.294514      4.556793
126324                  -1.000000  20160920     8.312564      4.521231
112727                  -1.000000  20161120     8.358855      4.569348
87498                   -1.000000  20170225     9.381663      8.754902
168266                  -1.000000  20170209    10.429274     13.858520
213458                  -1.000000  20170128     8.316288      4.481970
137337                  -1.000000  20160713     8.379169      4.575227
54886                   -1.000000  20160620     8.321066      4.636960
207892                  -1.000000  20170406     8.282690      4.542300
110268                  -1.000000  20160422     8.348518      4.502573
119879                  -1.000000  20161210    10.438691     13.924962
103694                  -1.000000  20170217     8.349981      4.501537
131932                  -1.000000  20170322     9.315270      8.763373
146867                  -1.000000  20160901     9.344401      8.787340
121958                  -1.000000  20170109    10.701345     13.765025

        lon_plus_lat  air_store_id2
107651    175.397389            568
163691    169.598128             88
92590     172.436835            539
226442    175.397389            107
28716     166.841262            748
139342    170.280342             72
24184     169.598128            186
78564     175.365828            427
34156     166.841262              7
186975    163.982029            686
228618    175.336004            326
51795     170.394193             83
200350    167.848212            320
55685     175.365828            554
81228     175.299819            471
112227    176.952694            483
228229    175.336004            326
2010      175.409667            819
32564     175.590754            267
165237    163.769876            584
124473    169.500502            497
186348    163.982029             11
111985    186.135454            646
47559     170.280342            151
190380    175.336004            514
198074    175.299819            808
171841    175.121857             23
180422    179.139479            525
61667     175.492603            437
103188    170.215387            411
...              ...            ...
184779    163.752875             22
214176    175.278569             30
235796    170.191062            193
103355    170.215387            411
5311      175.409667            425
199041    175.442723            256
64925     175.492603            793
194027    163.982029            548
59735     163.982029             79
769       170.215387            446
64820     167.848212            634
67221     175.409667            419
41090     175.299819             76
16023     175.339634            573
191335    175.365828            105
175203    175.442723             32
126324    175.460235            745
112727    175.365828            355
87498     170.157465            507
168266    164.006236            733
213458    175.495773            769
137337    175.339634            512
54886     175.336004            751
207892    175.469040            635
110268    175.442940            124
119879    163.930377            213
103694    175.442512            738
131932    170.215387            347
146867    170.162290            565
121958    163.827660            428

[201686 rows x 50 columns]

In [61]: %paste
def get_final_parameters(best_params, origin_model_d, current_model):
    """
    Fix the string parameters returned by hyperopt.
    Use the index given by hyperopt to find the real string value
    for a specific parameter
    """
    for element_p in list(best_params.keys()):
        if isinstance(
                origin_model_d[current_model][1][element_p],
                hyperopt.pyll.base.Apply):
            if origin_model_d[current_model][1][element_p].name == "float":
                pass
            elif origin_model_d[current_model][1][element_p].name == "switch":
                apply_obj = origin_model_d[current_model][1][element_p]
                literal_obj = apply_obj.pos_args[best_params[element_p] + 1]
                best_params[element_p] = literal_obj.obj
            else:
                pass
        else:
            pass
    return best_params

def regression_params_opt(
        origin_model_d,
        current_model,
        X_train,
        X_test,
        y_train,
        y_test):
    print(current_model)
    best = 0
    global i
    i = 0
    max_eval = 10
    trials = Trials()

    def rmse_score(params):
        global i
        model_fit = origin_model_d[current_model][0](
            **params).fit(x_train, y_train,eval_set=[(x_train,y_train),(x_test,y_test),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(x_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        params['best_iteration'] = model_fit.best_iteration
        df_result_hyperopt.loc[i] = np.append(loss, list(params.values()))
        i = i + 1
        return {'loss': loss, 'status': STATUS_OK}

    def rmse_score_nicolas(params):
        global i
        print("origin_model_d[current_model]",origin_model_d[current_model][0](**params))
        v = np.array(range(x_train.shape[0]))%2==0
        #v = [e%2==0 for e in range(x_train.shape[0]]
        model_fit = origin_model_d[current_model][0](
            **params).fit(x_train[v], y_train[v],eval_set=[(x_train[v],y_train[v]),(x_train[v==False],y_train[v==False]),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(x_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        df_result_hyperopt.loc[i] = np.append(loss, list(params.values()))
        i = i + 1
        return {'loss': loss, 'status': STATUS_OK}

    df_result_hyperopt = pd.DataFrame(
        columns=[
            np.append(
                'score', list(
                    origin_model_d[current_model][1].keys()))])

    best = fmin(rmse_score,
                origin_model_d[current_model][1],
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials)
    print(best)
    print(best.keys())
    print("\n")
    return get_final_parameters(best, origin_model_d, current_model)

if __name__ == '__main__':

    xgbr_d = {'gamma': hp.quniform('gamma', 0.0, 5.0, 0.1),
              'learning_rate': hp.choice('learning_rate', [0.1]),
              'colsample_bytree': hp.quniform('colsample_bytree',
                                              0.3,
                                              1.,
                                              0.05),
              'max_depth': hp.choice('max_depth', list(range(5, 10))),
              'min_child_weight': hp.quniform('min_child_weight', 1., 5., 1),
              'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
              'nthread': hp.choice('nthread', [-1]),
              'n_estimators': hp.choice('n_estimators', [1500]),
              'objective': hp.choice('objective', ['reg:linear']),
              'reg_lambda': hp.quniform('reg_alpha', 0.0, 4.0, 0.1),
              'reg_alpha': hp.quniform('reg_lambda', 0.0, 4.0, 0.1)}

    base_model = {"XGBRegressor": [XGBRegressor, xgbr_d]}


    #x_train = train[col]
    #y_train = train["visitors"]

    #x_test = test[col]
    #y_test = test["visitors"]

    hyper_parametres = regression_params_opt(
        origin_model_d=base_model,
        current_model="XGBRegressor",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

aaa = df_result_hyperopt[:]

df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")
## -- End pasted text --
XGBRegressor
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-61-1dc4aad51f92> in <module>()
    107         X_test=X_test,
    108         y_train=y_train,
--> 109         y_test=y_test
    110     )
    111

<ipython-input-61-1dc4aad51f92> in regression_params_opt(origin_model_d, current_model, X_train, X_test, y_train, y_test)
     69                 algo=tpe.suggest,
     70                 max_evals=max_eval,
---> 71                 trials=trials)
     72     print(best)
     73     print(best.keys())

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    305             verbose=verbose,
    306             catch_eval_exceptions=catch_eval_exceptions,
--> 307             return_argmin=return_argmin,
    308         )
    309

~/python3/lib/python3.6/site-packages/hyperopt/base.py in fmin(self, fn, space, algo, max_evals, rstate, verbose, pass_expr_memo_ctrl, catch_eval_exceptions, return_argmin)
    633             pass_expr_memo_ctrl=pass_expr_memo_ctrl,
    634             catch_eval_exceptions=catch_eval_exceptions,
--> 635             return_argmin=return_argmin)
    636
    637

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    318                     verbose=verbose)
    319     rval.catch_eval_exceptions = catch_eval_exceptions
--> 320     rval.exhaust()
    321     if return_argmin:
    322         return trials.argmin

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in exhaust(self)
    197     def exhaust(self):
    198         n_done = len(self.trials)
--> 199         self.run(self.max_evals - n_done, block_until_done=self.async)
    200         self.trials.refresh()
    201         return self

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in run(self, N, block_until_done)
    171             else:
    172                 # -- loop over trials and do the jobs directly
--> 173                 self.serial_evaluate()
    174
    175             if stopped:

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in serial_evaluate(self, N)
     90                 ctrl = base.Ctrl(self.trials, current_trial=trial)
     91                 try:
---> 92                     result = self.domain.evaluate(spec, ctrl)
     93                 except Exception as e:
     94                     logger.info('job exception: %s' % str(e))

~/python3/lib/python3.6/site-packages/hyperopt/base.py in evaluate(self, config, ctrl, attach_attachments)
    838                 memo=memo,
    839                 print_node_on_error=self.rec_eval_print_node_on_error)
--> 840             rval = self.fn(pyll_rval)
    841
    842         if isinstance(rval, (float, int, np.number)):

<ipython-input-61-1dc4aad51f92> in rmse_score(params)
     38         global i
     39         model_fit = origin_model_d[current_model][0](
---> 40             **params).fit(x_train, y_train,eval_set=[(x_train,y_train),(x_test,y_test),],early_stopping_rounds=50)
     41         y_pred_train = model_fit.predict(x_test)
     42         loss = mean_squared_error(y_test, y_pred_train)**0.5

NameError: name 'x_train' is not defined

In [62]: %paste
def get_final_parameters(best_params, origin_model_d, current_model):
    """
    Fix the string parameters returned by hyperopt.
    Use the index given by hyperopt to find the real string value
    for a specific parameter
    """
    for element_p in list(best_params.keys()):
        if isinstance(
                origin_model_d[current_model][1][element_p],
                hyperopt.pyll.base.Apply):
            if origin_model_d[current_model][1][element_p].name == "float":
                pass
            elif origin_model_d[current_model][1][element_p].name == "switch":
                apply_obj = origin_model_d[current_model][1][element_p]
                literal_obj = apply_obj.pos_args[best_params[element_p] + 1]
                best_params[element_p] = literal_obj.obj
            else:
                pass
        else:
            pass
    return best_params

def regression_params_opt(
        origin_model_d,
        current_model,
        X_train,
        X_test,
        y_train,
        y_test):
    print(current_model)
    best = 0
    global i
    i = 0
    max_eval = 10
    trials = Trials()

    def rmse_score(params):
        global i
        model_fit = origin_model_d[current_model][0](
            **params).fit(X_train, y_train,eval_set=[(X_train,y_train),(X_test,y_test),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(X_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        params['best_iteration'] = model_fit.best_iteration
        df_result_hyperopt.loc[i] = np.append(loss, list(params.values()))
        i = i + 1
        return {'loss': loss, 'status': STATUS_OK}

    def rmse_score_nicolas(params):
        global i
        print("origin_model_d[current_model]",origin_model_d[current_model][0](**params))
        v = np.array(range(x_train.shape[0]))%2==0
        #v = [e%2==0 for e in range(x_train.shape[0]]
        model_fit = origin_model_d[current_model][0](
            **params).fit(x_train[v], y_train[v],eval_set=[(x_train[v],y_train[v]),(x_train[v==False],y_train[v==False]),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(x_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        df_result_hyperopt.loc[i] = np.append(loss, list(params.values()))
        i = i + 1
        return {'loss': loss, 'status': STATUS_OK}

    df_result_hyperopt = pd.DataFrame(
        columns=[
            np.append(
                'score', list(
                    origin_model_d[current_model][1].keys()))])

    best = fmin(rmse_score,
                origin_model_d[current_model][1],
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials)
    print(best)
    print(best.keys())
    print("\n")
    return get_final_parameters(best, origin_model_d, current_model)

if __name__ == '__main__':

    xgbr_d = {'gamma': hp.quniform('gamma', 0.0, 5.0, 0.1),
              'learning_rate': hp.choice('learning_rate', [0.1]),
              'colsample_bytree': hp.quniform('colsample_bytree',
                                              0.3,
                                              1.,
                                              0.05),
              'max_depth': hp.choice('max_depth', list(range(5, 10))),
              'min_child_weight': hp.quniform('min_child_weight', 1., 5., 1),
              'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
              'nthread': hp.choice('nthread', [-1]),
              'n_estimators': hp.choice('n_estimators', [1500]),
              'objective': hp.choice('objective', ['reg:linear']),
              'reg_lambda': hp.quniform('reg_alpha', 0.0, 4.0, 0.1),
              'reg_alpha': hp.quniform('reg_lambda', 0.0, 4.0, 0.1)}

    base_model = {"XGBRegressor": [XGBRegressor, xgbr_d]}


    #x_train = train[col]
    #y_train = train["visitors"]

    #x_test = test[col]
    #y_test = test["visitors"]

    hyper_parametres = regression_params_opt(
        origin_model_d=base_model,
        current_model="XGBRegressor",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

aaa = df_result_hyperopt[:]

df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")
## -- End pasted text --
XGBRegressor
[0]	validation_0-rmse:24.2021	validation_1-rmse:24.5307
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.2592	validation_1-rmse:22.6062
[2]	validation_0-rmse:20.5382	validation_1-rmse:20.9174
[3]	validation_0-rmse:19.027	validation_1-rmse:19.4415
[4]	validation_0-rmse:17.7096	validation_1-rmse:18.1623
[5]	validation_0-rmse:16.5627	validation_1-rmse:17.0471
[6]	validation_0-rmse:15.562	validation_1-rmse:16.0883
[7]	validation_0-rmse:14.7098	validation_1-rmse:15.2672
[8]	validation_0-rmse:13.9828	validation_1-rmse:14.5733
[9]	validation_0-rmse:13.3583	validation_1-rmse:13.9808
[10]	validation_0-rmse:12.8104	validation_1-rmse:13.4712
[11]	validation_0-rmse:12.3552	validation_1-rmse:13.0486
[12]	validation_0-rmse:11.961	validation_1-rmse:12.6897
[13]	validation_0-rmse:11.599	validation_1-rmse:12.3876
[14]	validation_0-rmse:11.321	validation_1-rmse:12.1414
[15]	validation_0-rmse:11.0935	validation_1-rmse:11.939
[16]	validation_0-rmse:10.8855	validation_1-rmse:11.7661
[17]	validation_0-rmse:10.7248	validation_1-rmse:11.6235
[18]	validation_0-rmse:10.5718	validation_1-rmse:11.5009
[19]	validation_0-rmse:10.4524	validation_1-rmse:11.4038
[20]	validation_0-rmse:10.3543	validation_1-rmse:11.3206
[21]	validation_0-rmse:10.2584	validation_1-rmse:11.2546
[22]	validation_0-rmse:10.1846	validation_1-rmse:11.1998
[23]	validation_0-rmse:10.1005	validation_1-rmse:11.1593
[24]	validation_0-rmse:10.0259	validation_1-rmse:11.114
[25]	validation_0-rmse:9.94468	validation_1-rmse:11.0758
[26]	validation_0-rmse:9.90151	validation_1-rmse:11.0483
[27]	validation_0-rmse:9.8611	validation_1-rmse:11.026
[28]	validation_0-rmse:9.7924	validation_1-rmse:11.0177
[29]	validation_0-rmse:9.76117	validation_1-rmse:10.9989
[30]	validation_0-rmse:9.72697	validation_1-rmse:10.9829
[31]	validation_0-rmse:9.70344	validation_1-rmse:10.9695
[32]	validation_0-rmse:9.67612	validation_1-rmse:10.9569
[33]	validation_0-rmse:9.64201	validation_1-rmse:10.9463
[34]	validation_0-rmse:9.62363	validation_1-rmse:10.937
[35]	validation_0-rmse:9.59789	validation_1-rmse:10.9289
[36]	validation_0-rmse:9.58281	validation_1-rmse:10.9212
[37]	validation_0-rmse:9.57073	validation_1-rmse:10.9162
[38]	validation_0-rmse:9.54593	validation_1-rmse:10.9109
[39]	validation_0-rmse:9.53506	validation_1-rmse:10.9077
[40]	validation_0-rmse:9.50018	validation_1-rmse:10.9022
[41]	validation_0-rmse:9.48223	validation_1-rmse:10.9019
[42]	validation_0-rmse:9.45749	validation_1-rmse:10.9011
[43]	validation_0-rmse:9.44836	validation_1-rmse:10.8979
[44]	validation_0-rmse:9.43942	validation_1-rmse:10.8966
[45]	validation_0-rmse:9.43166	validation_1-rmse:10.8956
[46]	validation_0-rmse:9.4136	validation_1-rmse:10.8918
[47]	validation_0-rmse:9.4059	validation_1-rmse:10.8878
[48]	validation_0-rmse:9.37746	validation_1-rmse:10.8832
[49]	validation_0-rmse:9.35821	validation_1-rmse:10.8761
[50]	validation_0-rmse:9.34721	validation_1-rmse:10.8739
[51]	validation_0-rmse:9.32342	validation_1-rmse:10.8728
[52]	validation_0-rmse:9.30712	validation_1-rmse:10.8711
[53]	validation_0-rmse:9.29801	validation_1-rmse:10.869
[54]	validation_0-rmse:9.27906	validation_1-rmse:10.8637
[55]	validation_0-rmse:9.26394	validation_1-rmse:10.8612
[56]	validation_0-rmse:9.24285	validation_1-rmse:10.857
[57]	validation_0-rmse:9.23639	validation_1-rmse:10.8551
[58]	validation_0-rmse:9.23277	validation_1-rmse:10.8557
[59]	validation_0-rmse:9.22441	validation_1-rmse:10.8521
[60]	validation_0-rmse:9.21127	validation_1-rmse:10.8506
[61]	validation_0-rmse:9.20622	validation_1-rmse:10.8508
[62]	validation_0-rmse:9.1997	validation_1-rmse:10.8485
[63]	validation_0-rmse:9.1926	validation_1-rmse:10.8476
[64]	validation_0-rmse:9.18397	validation_1-rmse:10.8454
[65]	validation_0-rmse:9.17424	validation_1-rmse:10.8422
[66]	validation_0-rmse:9.15069	validation_1-rmse:10.8397
[67]	validation_0-rmse:9.14462	validation_1-rmse:10.8379
[68]	validation_0-rmse:9.13688	validation_1-rmse:10.8367
[69]	validation_0-rmse:9.11119	validation_1-rmse:10.8305
[70]	validation_0-rmse:9.10166	validation_1-rmse:10.8288
[71]	validation_0-rmse:9.09266	validation_1-rmse:10.8285
[72]	validation_0-rmse:9.08103	validation_1-rmse:10.8246
[73]	validation_0-rmse:9.07503	validation_1-rmse:10.8247
[74]	validation_0-rmse:9.03978	validation_1-rmse:10.8246
[75]	validation_0-rmse:9.03079	validation_1-rmse:10.8288
[76]	validation_0-rmse:9.0091	validation_1-rmse:10.8321
[77]	validation_0-rmse:8.99341	validation_1-rmse:10.8295
[78]	validation_0-rmse:8.98709	validation_1-rmse:10.8288
[79]	validation_0-rmse:8.98686	validation_1-rmse:10.8286
[80]	validation_0-rmse:8.98063	validation_1-rmse:10.8271
[81]	validation_0-rmse:8.97452	validation_1-rmse:10.8266
[82]	validation_0-rmse:8.95759	validation_1-rmse:10.8245
[83]	validation_0-rmse:8.95044	validation_1-rmse:10.8236
[84]	validation_0-rmse:8.9436	validation_1-rmse:10.823
[85]	validation_0-rmse:8.93488	validation_1-rmse:10.8239
[86]	validation_0-rmse:8.928	validation_1-rmse:10.8216
[87]	validation_0-rmse:8.92239	validation_1-rmse:10.8196
[88]	validation_0-rmse:8.91608	validation_1-rmse:10.8182
[89]	validation_0-rmse:8.8903	validation_1-rmse:10.8135
[90]	validation_0-rmse:8.88463	validation_1-rmse:10.8119
[91]	validation_0-rmse:8.86793	validation_1-rmse:10.8106
[92]	validation_0-rmse:8.86147	validation_1-rmse:10.8083
[93]	validation_0-rmse:8.85247	validation_1-rmse:10.811
[94]	validation_0-rmse:8.84476	validation_1-rmse:10.8094
[95]	validation_0-rmse:8.84209	validation_1-rmse:10.809
[96]	validation_0-rmse:8.83807	validation_1-rmse:10.807
[97]	validation_0-rmse:8.83293	validation_1-rmse:10.8058
[98]	validation_0-rmse:8.82587	validation_1-rmse:10.8078
[99]	validation_0-rmse:8.82571	validation_1-rmse:10.8078
[100]	validation_0-rmse:8.8202	validation_1-rmse:10.8077
[101]	validation_0-rmse:8.81347	validation_1-rmse:10.8066
[102]	validation_0-rmse:8.81039	validation_1-rmse:10.8061
[103]	validation_0-rmse:8.80794	validation_1-rmse:10.8058
[104]	validation_0-rmse:8.80293	validation_1-rmse:10.8046
[105]	validation_0-rmse:8.79331	validation_1-rmse:10.8005
[106]	validation_0-rmse:8.79208	validation_1-rmse:10.8002
[107]	validation_0-rmse:8.78844	validation_1-rmse:10.8005
[108]	validation_0-rmse:8.78471	validation_1-rmse:10.8019
[109]	validation_0-rmse:8.77988	validation_1-rmse:10.8036
[110]	validation_0-rmse:8.77799	validation_1-rmse:10.8029
[111]	validation_0-rmse:8.77789	validation_1-rmse:10.8029
[112]	validation_0-rmse:8.77117	validation_1-rmse:10.8023
[113]	validation_0-rmse:8.76745	validation_1-rmse:10.8009
[114]	validation_0-rmse:8.76737	validation_1-rmse:10.8009
[115]	validation_0-rmse:8.76197	validation_1-rmse:10.801
[116]	validation_0-rmse:8.75041	validation_1-rmse:10.7954
[117]	validation_0-rmse:8.75035	validation_1-rmse:10.7954
[118]	validation_0-rmse:8.75029	validation_1-rmse:10.7954
[119]	validation_0-rmse:8.75025	validation_1-rmse:10.7954
[120]	validation_0-rmse:8.73614	validation_1-rmse:10.7986
[121]	validation_0-rmse:8.73406	validation_1-rmse:10.7985
[122]	validation_0-rmse:8.72188	validation_1-rmse:10.7967
[123]	validation_0-rmse:8.72185	validation_1-rmse:10.7967
[124]	validation_0-rmse:8.72181	validation_1-rmse:10.7966
[125]	validation_0-rmse:8.72179	validation_1-rmse:10.7967
[126]	validation_0-rmse:8.71545	validation_1-rmse:10.7961
[127]	validation_0-rmse:8.71543	validation_1-rmse:10.7961
[128]	validation_0-rmse:8.71541	validation_1-rmse:10.7961
[129]	validation_0-rmse:8.7154	validation_1-rmse:10.7961
[130]	validation_0-rmse:8.71095	validation_1-rmse:10.795
[131]	validation_0-rmse:8.71026	validation_1-rmse:10.795
[132]	validation_0-rmse:8.70691	validation_1-rmse:10.7947
[133]	validation_0-rmse:8.70689	validation_1-rmse:10.7947
[134]	validation_0-rmse:8.70688	validation_1-rmse:10.7947
[135]	validation_0-rmse:8.70687	validation_1-rmse:10.7947
[136]	validation_0-rmse:8.70366	validation_1-rmse:10.7943
[137]	validation_0-rmse:8.70365	validation_1-rmse:10.7943
[138]	validation_0-rmse:8.70364	validation_1-rmse:10.7944
[139]	validation_0-rmse:8.69846	validation_1-rmse:10.7949
[140]	validation_0-rmse:8.6967	validation_1-rmse:10.7946
[141]	validation_0-rmse:8.6967	validation_1-rmse:10.7946
[142]	validation_0-rmse:8.6953	validation_1-rmse:10.7954
[143]	validation_0-rmse:8.69529	validation_1-rmse:10.7954
[144]	validation_0-rmse:8.69529	validation_1-rmse:10.7954
[145]	validation_0-rmse:8.69528	validation_1-rmse:10.7954
[146]	validation_0-rmse:8.69528	validation_1-rmse:10.7954
[147]	validation_0-rmse:8.69528	validation_1-rmse:10.7954
[148]	validation_0-rmse:8.69527	validation_1-rmse:10.7954
[149]	validation_0-rmse:8.69527	validation_1-rmse:10.7954
[150]	validation_0-rmse:8.69061	validation_1-rmse:10.795
[151]	validation_0-rmse:8.6906	validation_1-rmse:10.795
[152]	validation_0-rmse:8.6906	validation_1-rmse:10.795
[153]	validation_0-rmse:8.68155	validation_1-rmse:10.7936
[154]	validation_0-rmse:8.68155	validation_1-rmse:10.7936
[155]	validation_0-rmse:8.68155	validation_1-rmse:10.7936
[156]	validation_0-rmse:8.68155	validation_1-rmse:10.7936
[157]	validation_0-rmse:8.68155	validation_1-rmse:10.7936
[158]	validation_0-rmse:8.68154	validation_1-rmse:10.7936
[159]	validation_0-rmse:8.68154	validation_1-rmse:10.7936
[160]	validation_0-rmse:8.68154	validation_1-rmse:10.7936
[161]	validation_0-rmse:8.68154	validation_1-rmse:10.7936
[162]	validation_0-rmse:8.67884	validation_1-rmse:10.7953
[163]	validation_0-rmse:8.67739	validation_1-rmse:10.7938
[164]	validation_0-rmse:8.67739	validation_1-rmse:10.7938
[165]	validation_0-rmse:8.67738	validation_1-rmse:10.7938
[166]	validation_0-rmse:8.67207	validation_1-rmse:10.7954
[167]	validation_0-rmse:8.66707	validation_1-rmse:10.7959
[168]	validation_0-rmse:8.66707	validation_1-rmse:10.7959
[169]	validation_0-rmse:8.65811	validation_1-rmse:10.794
[170]	validation_0-rmse:8.65696	validation_1-rmse:10.7942
[171]	validation_0-rmse:8.65695	validation_1-rmse:10.7942
[172]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[173]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[174]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[175]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[176]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[177]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[178]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[179]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[180]	validation_0-rmse:8.65399	validation_1-rmse:10.7943
[181]	validation_0-rmse:8.64638	validation_1-rmse:10.794
[182]	validation_0-rmse:8.64638	validation_1-rmse:10.794
[183]	validation_0-rmse:8.64323	validation_1-rmse:10.7933
[184]	validation_0-rmse:8.64323	validation_1-rmse:10.7933
[185]	validation_0-rmse:8.63852	validation_1-rmse:10.7944
[186]	validation_0-rmse:8.63852	validation_1-rmse:10.7944
[187]	validation_0-rmse:8.63852	validation_1-rmse:10.7944
[188]	validation_0-rmse:8.63852	validation_1-rmse:10.7944
[189]	validation_0-rmse:8.63487	validation_1-rmse:10.7943
[190]	validation_0-rmse:8.63487	validation_1-rmse:10.7943
[191]	validation_0-rmse:8.6306	validation_1-rmse:10.7943
[192]	validation_0-rmse:8.6306	validation_1-rmse:10.7943
[193]	validation_0-rmse:8.6306	validation_1-rmse:10.7943
[194]	validation_0-rmse:8.6306	validation_1-rmse:10.7943
[195]	validation_0-rmse:8.6306	validation_1-rmse:10.7943
[196]	validation_0-rmse:8.6306	validation_1-rmse:10.7943
[197]	validation_0-rmse:8.62678	validation_1-rmse:10.7939
[198]	validation_0-rmse:8.62678	validation_1-rmse:10.7939
[199]	validation_0-rmse:8.62035	validation_1-rmse:10.7938
[200]	validation_0-rmse:8.62035	validation_1-rmse:10.7938
[201]	validation_0-rmse:8.62035	validation_1-rmse:10.7938
[202]	validation_0-rmse:8.62035	validation_1-rmse:10.7938
[203]	validation_0-rmse:8.61428	validation_1-rmse:10.7938
[204]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[205]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[206]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[207]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[208]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[209]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[210]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[211]	validation_0-rmse:8.60862	validation_1-rmse:10.792
[212]	validation_0-rmse:8.60746	validation_1-rmse:10.7925
[213]	validation_0-rmse:8.60746	validation_1-rmse:10.7925
[214]	validation_0-rmse:8.60319	validation_1-rmse:10.7937
[215]	validation_0-rmse:8.60319	validation_1-rmse:10.7937
[216]	validation_0-rmse:8.60319	validation_1-rmse:10.7937
[217]	validation_0-rmse:8.60319	validation_1-rmse:10.7936
[218]	validation_0-rmse:8.60319	validation_1-rmse:10.7937
[219]	validation_0-rmse:8.60319	validation_1-rmse:10.7936
[220]	validation_0-rmse:8.60319	validation_1-rmse:10.7936
[221]	validation_0-rmse:8.60319	validation_1-rmse:10.7936
[222]	validation_0-rmse:8.60319	validation_1-rmse:10.7936
[223]	validation_0-rmse:8.59821	validation_1-rmse:10.7941
[224]	validation_0-rmse:8.59821	validation_1-rmse:10.7941
[225]	validation_0-rmse:8.59609	validation_1-rmse:10.7931
[226]	validation_0-rmse:8.59609	validation_1-rmse:10.7931
[227]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[228]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[229]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[230]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[231]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[232]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[233]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[234]	validation_0-rmse:8.59514	validation_1-rmse:10.7929
[235]	validation_0-rmse:8.59261	validation_1-rmse:10.7932
[236]	validation_0-rmse:8.58638	validation_1-rmse:10.7924
[237]	validation_0-rmse:8.58638	validation_1-rmse:10.7924
[238]	validation_0-rmse:8.58638	validation_1-rmse:10.7924
[239]	validation_0-rmse:8.58638	validation_1-rmse:10.7924
[240]	validation_0-rmse:8.58638	validation_1-rmse:10.7924
[241]	validation_0-rmse:8.58005	validation_1-rmse:10.7923
[242]	validation_0-rmse:8.5768	validation_1-rmse:10.7932
[243]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[244]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[245]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[246]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[247]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[248]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[249]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[250]	validation_0-rmse:8.5687	validation_1-rmse:10.7973
[251]	validation_0-rmse:8.56674	validation_1-rmse:10.7974
[252]	validation_0-rmse:8.56674	validation_1-rmse:10.7974
[253]	validation_0-rmse:8.56486	validation_1-rmse:10.7971
[254]	validation_0-rmse:8.55638	validation_1-rmse:10.7966
[255]	validation_0-rmse:8.55638	validation_1-rmse:10.7966
[256]	validation_0-rmse:8.55638	validation_1-rmse:10.7966
[257]	validation_0-rmse:8.55638	validation_1-rmse:10.7966
[258]	validation_0-rmse:8.55448	validation_1-rmse:10.7972
[259]	validation_0-rmse:8.55448	validation_1-rmse:10.7972
Stopping. Best iteration:
[209]	validation_0-rmse:8.60862	validation_1-rmse:10.792

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-62-80a10551fb4d> in <module>()
    107         X_test=X_test,
    108         y_train=y_train,
--> 109         y_test=y_test
    110     )
    111

<ipython-input-62-80a10551fb4d> in regression_params_opt(origin_model_d, current_model, X_train, X_test, y_train, y_test)
     69                 algo=tpe.suggest,
     70                 max_evals=max_eval,
---> 71                 trials=trials)
     72     print(best)
     73     print(best.keys())

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    305             verbose=verbose,
    306             catch_eval_exceptions=catch_eval_exceptions,
--> 307             return_argmin=return_argmin,
    308         )
    309

~/python3/lib/python3.6/site-packages/hyperopt/base.py in fmin(self, fn, space, algo, max_evals, rstate, verbose, pass_expr_memo_ctrl, catch_eval_exceptions, return_argmin)
    633             pass_expr_memo_ctrl=pass_expr_memo_ctrl,
    634             catch_eval_exceptions=catch_eval_exceptions,
--> 635             return_argmin=return_argmin)
    636
    637

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    318                     verbose=verbose)
    319     rval.catch_eval_exceptions = catch_eval_exceptions
--> 320     rval.exhaust()
    321     if return_argmin:
    322         return trials.argmin

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in exhaust(self)
    197     def exhaust(self):
    198         n_done = len(self.trials)
--> 199         self.run(self.max_evals - n_done, block_until_done=self.async)
    200         self.trials.refresh()
    201         return self

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in run(self, N, block_until_done)
    171             else:
    172                 # -- loop over trials and do the jobs directly
--> 173                 self.serial_evaluate()
    174
    175             if stopped:

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in serial_evaluate(self, N)
     90                 ctrl = base.Ctrl(self.trials, current_trial=trial)
     91                 try:
---> 92                     result = self.domain.evaluate(spec, ctrl)
     93                 except Exception as e:
     94                     logger.info('job exception: %s' % str(e))

~/python3/lib/python3.6/site-packages/hyperopt/base.py in evaluate(self, config, ctrl, attach_attachments)
    838                 memo=memo,
    839                 print_node_on_error=self.rec_eval_print_node_on_error)
--> 840             rval = self.fn(pyll_rval)
    841
    842         if isinstance(rval, (float, int, np.number)):

<ipython-input-62-80a10551fb4d> in rmse_score(params)
     42         loss = mean_squared_error(y_test, y_pred_train)**0.5
     43         params['best_iteration'] = model_fit.best_iteration
---> 44         df_result_hyperopt.loc[i] = np.append(loss, list(params.values()))
     45         i = i + 1
     46         return {'loss': loss, 'status': STATUS_OK}

~/python3/lib/python3.6/site-packages/pandas/core/indexing.py in __setitem__(self, key, value)
    177             key = com._apply_if_callable(key, self.obj)
    178         indexer = self._get_setitem_indexer(key)
--> 179         self._setitem_with_indexer(indexer, value)
    180
    181     def _has_valid_type(self, k, axis):

~/python3/lib/python3.6/site-packages/pandas/core/indexing.py in _setitem_with_indexer(self, indexer, value)
    417                         if is_list_like_indexer(value):
    418                             if len(value) != len(self.obj.columns):
--> 419                                 raise ValueError("cannot set a row with "
    420                                                  "mismatched columns")
    421

ValueError: cannot set a row with mismatched columns

In [63]: y_test.shape
Out[63]: (50422,)

In [64]: X_test.shape
Out[64]: (50422, 50)

In [65]: X_train.shape
Out[65]: (201686, 50)

In [66]: y_train.shape
Out[66]: (201686,)

In [67]: y_test
Out[67]:
160503    33
74437     20
93611     24
36756     16
61315     10
99936     42
189042    48
78456      9
180807    11
47721     12
24393     69
5314      28
250320     8
25414     80
200723    54
189985    18
85254      4
139207    15
233812     2
17500     18
169610    21
78902     33
158608    36
216634    26
34823     41
121895    14
114732    44
203343    30
5015      35
46868     33
          ..
245562     4
101068     9
58875     12
209128    24
11060      4
159364    30
224081    11
82479      6
117361    16
54081     18
56463     18
244968    32
163280     4
5777       5
23749     26
16298     22
17917     22
36211     23
213738    60
76053     16
104373    22
93970     47
12586     15
118344    14
127825     7
129816    28
22786     37
170394    24
214967    18
149756     1
Name: visitors, Length: 50422, dtype: int64

In [68]: y_test.columns
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-68-d0d25bfdfeb7> in <module>()
----> 1 y_test.columns

~/python3/lib/python3.6/site-packages/pandas/core/generic.py in __getattr__(self, name)
   3075         if (name in self._internal_names_set or name in self._metadata or
   3076                 name in self._accessors):
-> 3077             return object.__getattribute__(self, name)
   3078         else:
   3079             if name in self._info_axis:

AttributeError: 'Series' object has no attribute 'columns'

In [69]: X_test.columns
Out[69]:
Index(['dow', 'year', 'month', 'day_of_week', 'holiday_flg', 'min_visitors',
       'mean_visitors', 'median_visitors', 'max_visitors',
       'count_observations', 'air_genre_name', 'air_area_name', 'latitude',
       'longitude', 'air_genre_name0', 'air_area_name0', 'air_genre_name1',
       'air_area_name1', 'air_genre_name2', 'air_area_name2',
       'air_genre_name3', 'air_area_name3', 'air_genre_name4',
       'air_area_name4', 'air_genre_name5', 'air_area_name5',
       'air_genre_name6', 'air_area_name6', 'air_genre_name7',
       'air_area_name7', 'air_genre_name8', 'air_area_name8',
       'air_genre_name9', 'air_area_name9', 'rs1_x', 'rv1_x', 'rs2_x', 'rv2_x',
       'rs1_y', 'rv1_y', 'rs2_y', 'rv2_y', 'total_reserv_sum',
       'total_reserv_mean', 'total_reserv_dt_diff_mean', 'date_int',
       'var_max_lat', 'var_max_long', 'lon_plus_lat', 'air_store_id2'],
      dtype='object')

In [70]: X_test.columns == X_train.colums
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-70-f2e4790f9f4f> in <module>()
----> 1 X_test.columns == X_train.colums

~/python3/lib/python3.6/site-packages/pandas/core/generic.py in __getattr__(self, name)
   3079             if name in self._info_axis:
   3080                 return self[name]
-> 3081             return object.__getattribute__(self, name)
   3082
   3083     def __setattr__(self, name, value):

AttributeError: 'DataFrame' object has no attribute 'colums'

In [71]: X_test.columns == X_train.columns
Out[71]:
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True], dtype=bool)

In [72]: df_result_hyperopt
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-72-4e4893b56831> in <module>()
----> 1 df_result_hyperopt

NameError: name 'df_result_hyperopt' is not defined

In [73]: %paste
def get_final_parameters(best_params, origin_model_d, current_model):
    """
    Fix the string parameters returned by hyperopt.
    Use the index given by hyperopt to find the real string value
    for a specific parameter
    """
    for element_p in list(best_params.keys()):
        if isinstance(
                origin_model_d[current_model][1][element_p],
                hyperopt.pyll.base.Apply):
            if origin_model_d[current_model][1][element_p].name == "float":
                pass
            elif origin_model_d[current_model][1][element_p].name == "switch":
                apply_obj = origin_model_d[current_model][1][element_p]
                literal_obj = apply_obj.pos_args[best_params[element_p] + 1]
                best_params[element_p] = literal_obj.obj
            else:
                pass
        else:
            pass
    return best_params

def regression_params_opt(
        origin_model_d,
        current_model,
        X_train,
        X_test,
        y_train,
        y_test):
    print(current_model)
    best = 0
    max_eval = 10
    trials = Trials()

    def rmse_score(params):
        model_fit = origin_model_d[current_model][0](
            **params).fit(X_train, y_train,eval_set=[(X_train,y_train),(X_test,y_test),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(X_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        params['best_iteration'] = model_fit.best_iteration
        list_result_hyperopt.append((loss, list(params.values())))
        return {'loss': loss, 'status': STATUS_OK}

    best = fmin(rmse_score,
                origin_model_d[current_model][1],
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials)
    print(best)
    print(best.keys())
    print("\n")
    return get_final_parameters(best, origin_model_d, current_model)

if __name__ == '__main__':

    xgbr_d = {'gamma': hp.quniform('gamma', 0.0, 5.0, 0.1),
              'learning_rate': hp.choice('learning_rate', [0.1]),
              'colsample_bytree': hp.quniform('colsample_bytree',
                                              0.3,
                                              1.,
                                              0.05),
              'max_depth': hp.choice('max_depth', list(range(5, 10))),
              'min_child_weight': hp.quniform('min_child_weight', 1., 5., 1),
              'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
              'nthread': hp.choice('nthread', [-1]),
              'n_estimators': hp.choice('n_estimators', [1500]),
              'objective': hp.choice('objective', ['reg:linear']),
              'reg_lambda': hp.quniform('reg_alpha', 0.0, 4.0, 0.1),
              'reg_alpha': hp.quniform('reg_lambda', 0.0, 4.0, 0.1)}

    base_model = {"XGBRegressor": [XGBRegressor, xgbr_d]}


    #x_train = train[col]
    #y_train = train["visitors"]

    #x_test = test[col]
    #y_test = test["visitors"]

    list_result_hyperopt = []
    hyper_parametres = regression_params_opt(
        origin_model_d=base_model,
        current_model="XGBRegressor",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

## -- End pasted text --
XGBRegressor
[0]	validation_0-rmse:24.2404	validation_1-rmse:24.5541
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3335	validation_1-rmse:22.652
[2]	validation_0-rmse:20.6564	validation_1-rmse:20.98
[3]	validation_0-rmse:19.3141	validation_1-rmse:19.6474
[4]	validation_0-rmse:18.1603	validation_1-rmse:18.5037
[5]	validation_0-rmse:17.1372	validation_1-rmse:17.4892
[6]	validation_0-rmse:16.1336	validation_1-rmse:16.4954
[7]	validation_0-rmse:15.3762	validation_1-rmse:15.7474
[8]	validation_0-rmse:14.6311	validation_1-rmse:15.0096
[9]	validation_0-rmse:13.9903	validation_1-rmse:14.3766
[10]	validation_0-rmse:13.4428	validation_1-rmse:13.8373
[11]	validation_0-rmse:12.9833	validation_1-rmse:13.3836
[12]	validation_0-rmse:12.602	validation_1-rmse:13.0087
[13]	validation_0-rmse:12.3668	validation_1-rmse:12.7805
[14]	validation_0-rmse:12.1278	validation_1-rmse:12.5473
[15]	validation_0-rmse:11.8849	validation_1-rmse:12.3098
[16]	validation_0-rmse:11.7124	validation_1-rmse:12.1399
[17]	validation_0-rmse:11.5378	validation_1-rmse:11.9693
[18]	validation_0-rmse:11.4219	validation_1-rmse:11.8565
[19]	validation_0-rmse:11.2947	validation_1-rmse:11.7322
[20]	validation_0-rmse:11.2141	validation_1-rmse:11.6545
[21]	validation_0-rmse:11.1176	validation_1-rmse:11.5682
[22]	validation_0-rmse:11.0675	validation_1-rmse:11.5215
[23]	validation_0-rmse:10.9934	validation_1-rmse:11.4542
[24]	validation_0-rmse:10.9333	validation_1-rmse:11.394
[25]	validation_0-rmse:10.8821	validation_1-rmse:11.3429
[26]	validation_0-rmse:10.84	validation_1-rmse:11.3008
[27]	validation_0-rmse:10.8049	validation_1-rmse:11.2638
[28]	validation_0-rmse:10.7773	validation_1-rmse:11.2425
[29]	validation_0-rmse:10.7508	validation_1-rmse:11.216
[30]	validation_0-rmse:10.7285	validation_1-rmse:11.1932
[31]	validation_0-rmse:10.7108	validation_1-rmse:11.1761
[32]	validation_0-rmse:10.6972	validation_1-rmse:11.1644
[33]	validation_0-rmse:10.6768	validation_1-rmse:11.1495
[34]	validation_0-rmse:10.6627	validation_1-rmse:11.1425
[35]	validation_0-rmse:10.6535	validation_1-rmse:11.1341
[36]	validation_0-rmse:10.6406	validation_1-rmse:11.1227
[37]	validation_0-rmse:10.6316	validation_1-rmse:11.1144
[38]	validation_0-rmse:10.6186	validation_1-rmse:11.1074
[39]	validation_0-rmse:10.6114	validation_1-rmse:11.1014
[40]	validation_0-rmse:10.6038	validation_1-rmse:11.0941
[41]	validation_0-rmse:10.5992	validation_1-rmse:11.0941
[42]	validation_0-rmse:10.5931	validation_1-rmse:11.0897
[43]	validation_0-rmse:10.5871	validation_1-rmse:11.0862
[44]	validation_0-rmse:10.5802	validation_1-rmse:11.0811
[45]	validation_0-rmse:10.574	validation_1-rmse:11.0798
[46]	validation_0-rmse:10.5644	validation_1-rmse:11.0725
[47]	validation_0-rmse:10.5564	validation_1-rmse:11.0651
[48]	validation_0-rmse:10.5492	validation_1-rmse:11.0666
[49]	validation_0-rmse:10.5437	validation_1-rmse:11.0632
[50]	validation_0-rmse:10.5377	validation_1-rmse:11.0667
[51]	validation_0-rmse:10.533	validation_1-rmse:11.0618
[52]	validation_0-rmse:10.529	validation_1-rmse:11.0612
[53]	validation_0-rmse:10.5253	validation_1-rmse:11.0594
[54]	validation_0-rmse:10.5191	validation_1-rmse:11.0589
[55]	validation_0-rmse:10.5168	validation_1-rmse:11.0568
[56]	validation_0-rmse:10.5154	validation_1-rmse:11.06
[57]	validation_0-rmse:10.5071	validation_1-rmse:11.0616
[58]	validation_0-rmse:10.504	validation_1-rmse:11.0599
[59]	validation_0-rmse:10.5011	validation_1-rmse:11.0582
[60]	validation_0-rmse:10.4962	validation_1-rmse:11.056
[61]	validation_0-rmse:10.4918	validation_1-rmse:11.0587
[62]	validation_0-rmse:10.4895	validation_1-rmse:11.0568
[63]	validation_0-rmse:10.4879	validation_1-rmse:11.0563
[64]	validation_0-rmse:10.4857	validation_1-rmse:11.0552
[65]	validation_0-rmse:10.4835	validation_1-rmse:11.0535
[66]	validation_0-rmse:10.4785	validation_1-rmse:11.0528
[67]	validation_0-rmse:10.4678	validation_1-rmse:11.0526
[68]	validation_0-rmse:10.4604	validation_1-rmse:11.05
[69]	validation_0-rmse:10.4538	validation_1-rmse:11.0502
[70]	validation_0-rmse:10.4472	validation_1-rmse:11.0505
[71]	validation_0-rmse:10.4432	validation_1-rmse:11.054
[72]	validation_0-rmse:10.4342	validation_1-rmse:11.052
[73]	validation_0-rmse:10.428	validation_1-rmse:11.0484
[74]	validation_0-rmse:10.4261	validation_1-rmse:11.0474
[75]	validation_0-rmse:10.4216	validation_1-rmse:11.0455
[76]	validation_0-rmse:10.4158	validation_1-rmse:11.0454
[77]	validation_0-rmse:10.4136	validation_1-rmse:11.0436
[78]	validation_0-rmse:10.4082	validation_1-rmse:11.0408
[79]	validation_0-rmse:10.4056	validation_1-rmse:11.0396
[80]	validation_0-rmse:10.403	validation_1-rmse:11.0382
[81]	validation_0-rmse:10.4008	validation_1-rmse:11.0368
[82]	validation_0-rmse:10.3969	validation_1-rmse:11.036
[83]	validation_0-rmse:10.3942	validation_1-rmse:11.0371
[84]	validation_0-rmse:10.3907	validation_1-rmse:11.0366
[85]	validation_0-rmse:10.387	validation_1-rmse:11.0338
[86]	validation_0-rmse:10.3838	validation_1-rmse:11.0317
[87]	validation_0-rmse:10.3826	validation_1-rmse:11.0313
[88]	validation_0-rmse:10.3776	validation_1-rmse:11.0326
[89]	validation_0-rmse:10.3768	validation_1-rmse:11.0322
[90]	validation_0-rmse:10.3734	validation_1-rmse:11.0291
[91]	validation_0-rmse:10.37	validation_1-rmse:11.0288
[92]	validation_0-rmse:10.3687	validation_1-rmse:11.0285
[93]	validation_0-rmse:10.3655	validation_1-rmse:11.0321
[94]	validation_0-rmse:10.3624	validation_1-rmse:11.032
[95]	validation_0-rmse:10.3614	validation_1-rmse:11.0314
[96]	validation_0-rmse:10.3583	validation_1-rmse:11.0295
[97]	validation_0-rmse:10.3553	validation_1-rmse:11.0269
[98]	validation_0-rmse:10.3524	validation_1-rmse:11.0254
[99]	validation_0-rmse:10.3508	validation_1-rmse:11.0245
[100]	validation_0-rmse:10.3481	validation_1-rmse:11.0234
[101]	validation_0-rmse:10.3462	validation_1-rmse:11.0229
[102]	validation_0-rmse:10.3435	validation_1-rmse:11.0206
[103]	validation_0-rmse:10.3418	validation_1-rmse:11.0201
[104]	validation_0-rmse:10.3331	validation_1-rmse:11.0207
[105]	validation_0-rmse:10.3321	validation_1-rmse:11.0203
[106]	validation_0-rmse:10.3283	validation_1-rmse:11.0205
[107]	validation_0-rmse:10.3229	validation_1-rmse:11.0218
[108]	validation_0-rmse:10.3219	validation_1-rmse:11.0207
[109]	validation_0-rmse:10.321	validation_1-rmse:11.0214
[110]	validation_0-rmse:10.3186	validation_1-rmse:11.0193
[111]	validation_0-rmse:10.318	validation_1-rmse:11.0213
[112]	validation_0-rmse:10.3167	validation_1-rmse:11.0212
[113]	validation_0-rmse:10.3157	validation_1-rmse:11.021
[114]	validation_0-rmse:10.3149	validation_1-rmse:11.0203
[115]	validation_0-rmse:10.3137	validation_1-rmse:11.021
[116]	validation_0-rmse:10.3106	validation_1-rmse:11.0189
[117]	validation_0-rmse:10.3058	validation_1-rmse:11.0177
[118]	validation_0-rmse:10.3042	validation_1-rmse:11.0174
[119]	validation_0-rmse:10.3031	validation_1-rmse:11.0166
[120]	validation_0-rmse:10.3019	validation_1-rmse:11.0164
[121]	validation_0-rmse:10.3004	validation_1-rmse:11.0159
[122]	validation_0-rmse:10.2988	validation_1-rmse:11.0144
[123]	validation_0-rmse:10.2956	validation_1-rmse:11.0126
[124]	validation_0-rmse:10.2885	validation_1-rmse:11.0131
[125]	validation_0-rmse:10.288	validation_1-rmse:11.0128
[126]	validation_0-rmse:10.2873	validation_1-rmse:11.0138
[127]	validation_0-rmse:10.2862	validation_1-rmse:11.013
[128]	validation_0-rmse:10.2857	validation_1-rmse:11.011
[129]	validation_0-rmse:10.2836	validation_1-rmse:11.0105
[130]	validation_0-rmse:10.2833	validation_1-rmse:11.0119
[131]	validation_0-rmse:10.2775	validation_1-rmse:11.0185
[132]	validation_0-rmse:10.2772	validation_1-rmse:11.02
[133]	validation_0-rmse:10.2729	validation_1-rmse:11.0246
[134]	validation_0-rmse:10.2715	validation_1-rmse:11.0293
[135]	validation_0-rmse:10.2688	validation_1-rmse:11.0284
[136]	validation_0-rmse:10.2675	validation_1-rmse:11.0281
[137]	validation_0-rmse:10.2632	validation_1-rmse:11.0338
[138]	validation_0-rmse:10.2566	validation_1-rmse:11.0386
[139]	validation_0-rmse:10.255	validation_1-rmse:11.0352
[140]	validation_0-rmse:10.2545	validation_1-rmse:11.0315
[141]	validation_0-rmse:10.253	validation_1-rmse:11.0317
[142]	validation_0-rmse:10.248	validation_1-rmse:11.0355
[143]	validation_0-rmse:10.2472	validation_1-rmse:11.0355
[144]	validation_0-rmse:10.2423	validation_1-rmse:11.0403
[145]	validation_0-rmse:10.2381	validation_1-rmse:11.0486
[146]	validation_0-rmse:10.233	validation_1-rmse:11.0485
[147]	validation_0-rmse:10.2319	validation_1-rmse:11.048
[148]	validation_0-rmse:10.2312	validation_1-rmse:11.0476
[149]	validation_0-rmse:10.229	validation_1-rmse:11.0459
[150]	validation_0-rmse:10.2282	validation_1-rmse:11.0471
[151]	validation_0-rmse:10.2273	validation_1-rmse:11.0465
[152]	validation_0-rmse:10.2248	validation_1-rmse:11.0447
[153]	validation_0-rmse:10.2237	validation_1-rmse:11.0441
[154]	validation_0-rmse:10.2223	validation_1-rmse:11.0461
[155]	validation_0-rmse:10.2184	validation_1-rmse:11.0457
[156]	validation_0-rmse:10.2183	validation_1-rmse:11.0412
[157]	validation_0-rmse:10.2175	validation_1-rmse:11.0409
[158]	validation_0-rmse:10.2148	validation_1-rmse:11.0397
[159]	validation_0-rmse:10.2142	validation_1-rmse:11.0401
[160]	validation_0-rmse:10.2138	validation_1-rmse:11.0398
[161]	validation_0-rmse:10.2121	validation_1-rmse:11.0392
[162]	validation_0-rmse:10.2091	validation_1-rmse:11.0388
[163]	validation_0-rmse:10.207	validation_1-rmse:11.0364
[164]	validation_0-rmse:10.2055	validation_1-rmse:11.0339
[165]	validation_0-rmse:10.2025	validation_1-rmse:11.0375
[166]	validation_0-rmse:10.202	validation_1-rmse:11.0388
[167]	validation_0-rmse:10.2017	validation_1-rmse:11.0388
[168]	validation_0-rmse:10.2	validation_1-rmse:11.0385
[169]	validation_0-rmse:10.1999	validation_1-rmse:11.0375
[170]	validation_0-rmse:10.1991	validation_1-rmse:11.0371
[171]	validation_0-rmse:10.1967	validation_1-rmse:11.0342
[172]	validation_0-rmse:10.1962	validation_1-rmse:11.0352
[173]	validation_0-rmse:10.1934	validation_1-rmse:11.0361
[174]	validation_0-rmse:10.1922	validation_1-rmse:11.0357
[175]	validation_0-rmse:10.192	validation_1-rmse:11.0307
[176]	validation_0-rmse:10.1896	validation_1-rmse:11.0296
[177]	validation_0-rmse:10.1884	validation_1-rmse:11.0289
[178]	validation_0-rmse:10.1854	validation_1-rmse:11.0317
[179]	validation_0-rmse:10.1846	validation_1-rmse:11.0321
Stopping. Best iteration:
[129]	validation_0-rmse:10.2836	validation_1-rmse:11.0105

[0]	validation_0-rmse:24.2311	validation_1-rmse:24.5495
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3152	validation_1-rmse:22.6437
[2]	validation_0-rmse:20.6291	validation_1-rmse:20.9685
[3]	validation_0-rmse:19.1509	validation_1-rmse:19.5053
[4]	validation_0-rmse:17.8603	validation_1-rmse:18.235
[5]	validation_0-rmse:16.7373	validation_1-rmse:17.1253
[6]	validation_0-rmse:15.7646	validation_1-rmse:16.168
[7]	validation_0-rmse:14.9287	validation_1-rmse:15.3481
[8]	validation_0-rmse:14.217	validation_1-rmse:14.6518
[9]	validation_0-rmse:13.6063	validation_1-rmse:14.054
[10]	validation_0-rmse:13.0753	validation_1-rmse:13.5436
[11]	validation_0-rmse:12.6321	validation_1-rmse:13.1207
[12]	validation_0-rmse:12.2513	validation_1-rmse:12.7614
[13]	validation_0-rmse:11.9286	validation_1-rmse:12.4612
[14]	validation_0-rmse:11.6585	validation_1-rmse:12.2135
[15]	validation_0-rmse:11.4397	validation_1-rmse:12.0068
[16]	validation_0-rmse:11.2427	validation_1-rmse:11.833
[17]	validation_0-rmse:11.0865	validation_1-rmse:11.6869
[18]	validation_0-rmse:10.9517	validation_1-rmse:11.5662
[19]	validation_0-rmse:10.8438	validation_1-rmse:11.4686
[20]	validation_0-rmse:10.7512	validation_1-rmse:11.3858
[21]	validation_0-rmse:10.6686	validation_1-rmse:11.3171
[22]	validation_0-rmse:10.5982	validation_1-rmse:11.2602
[23]	validation_0-rmse:10.537	validation_1-rmse:11.2145
[24]	validation_0-rmse:10.4803	validation_1-rmse:11.1726
[25]	validation_0-rmse:10.4261	validation_1-rmse:11.1392
[26]	validation_0-rmse:10.3898	validation_1-rmse:11.1113
[27]	validation_0-rmse:10.3548	validation_1-rmse:11.0925
[28]	validation_0-rmse:10.3197	validation_1-rmse:11.0743
[29]	validation_0-rmse:10.2912	validation_1-rmse:11.0524
[30]	validation_0-rmse:10.2622	validation_1-rmse:11.039
[31]	validation_0-rmse:10.2371	validation_1-rmse:11.0253
[32]	validation_0-rmse:10.2167	validation_1-rmse:11.0148
[33]	validation_0-rmse:10.2017	validation_1-rmse:11.0058
[34]	validation_0-rmse:10.1869	validation_1-rmse:10.9983
[35]	validation_0-rmse:10.164	validation_1-rmse:10.9942
[36]	validation_0-rmse:10.1512	validation_1-rmse:10.986
[37]	validation_0-rmse:10.1421	validation_1-rmse:10.9802
[38]	validation_0-rmse:10.1251	validation_1-rmse:10.9784
[39]	validation_0-rmse:10.1143	validation_1-rmse:10.972
[40]	validation_0-rmse:10.0971	validation_1-rmse:10.9662
[41]	validation_0-rmse:10.084	validation_1-rmse:10.9626
[42]	validation_0-rmse:10.0772	validation_1-rmse:10.9603
[43]	validation_0-rmse:10.0706	validation_1-rmse:10.9569
[44]	validation_0-rmse:10.0597	validation_1-rmse:10.9549
[45]	validation_0-rmse:10.048	validation_1-rmse:10.955
[46]	validation_0-rmse:10.0384	validation_1-rmse:10.9538
[47]	validation_0-rmse:10.0314	validation_1-rmse:10.9513
[48]	validation_0-rmse:10.0196	validation_1-rmse:10.9473
[49]	validation_0-rmse:10.0096	validation_1-rmse:10.9437
[50]	validation_0-rmse:10.0039	validation_1-rmse:10.9441
[51]	validation_0-rmse:9.98703	validation_1-rmse:10.9462
[52]	validation_0-rmse:9.9755	validation_1-rmse:10.9405
[53]	validation_0-rmse:9.973	validation_1-rmse:10.939
[54]	validation_0-rmse:9.96004	validation_1-rmse:10.9339
[55]	validation_0-rmse:9.94522	validation_1-rmse:10.9341
[56]	validation_0-rmse:9.93271	validation_1-rmse:10.9292
[57]	validation_0-rmse:9.91988	validation_1-rmse:10.9279
[58]	validation_0-rmse:9.91682	validation_1-rmse:10.9261
[59]	validation_0-rmse:9.91119	validation_1-rmse:10.9243
[60]	validation_0-rmse:9.90073	validation_1-rmse:10.9217
[61]	validation_0-rmse:9.89515	validation_1-rmse:10.9227
[62]	validation_0-rmse:9.89113	validation_1-rmse:10.9207
[63]	validation_0-rmse:9.88702	validation_1-rmse:10.921
[64]	validation_0-rmse:9.88262	validation_1-rmse:10.92
[65]	validation_0-rmse:9.87907	validation_1-rmse:10.919
[66]	validation_0-rmse:9.87356	validation_1-rmse:10.9192
[67]	validation_0-rmse:9.86957	validation_1-rmse:10.9183
[68]	validation_0-rmse:9.85163	validation_1-rmse:10.9173
[69]	validation_0-rmse:9.8479	validation_1-rmse:10.9154
[70]	validation_0-rmse:9.83486	validation_1-rmse:10.9125
[71]	validation_0-rmse:9.82851	validation_1-rmse:10.9158
[72]	validation_0-rmse:9.82371	validation_1-rmse:10.9133
[73]	validation_0-rmse:9.81488	validation_1-rmse:10.9127
[74]	validation_0-rmse:9.80784	validation_1-rmse:10.9107
[75]	validation_0-rmse:9.80296	validation_1-rmse:10.9156
[76]	validation_0-rmse:9.7965	validation_1-rmse:10.9114
[77]	validation_0-rmse:9.78985	validation_1-rmse:10.9107
[78]	validation_0-rmse:9.78359	validation_1-rmse:10.9089
[79]	validation_0-rmse:9.78103	validation_1-rmse:10.9076
[80]	validation_0-rmse:9.7755	validation_1-rmse:10.9055
[81]	validation_0-rmse:9.76693	validation_1-rmse:10.9046
[82]	validation_0-rmse:9.76355	validation_1-rmse:10.9049
[83]	validation_0-rmse:9.75795	validation_1-rmse:10.9077
[84]	validation_0-rmse:9.75405	validation_1-rmse:10.907
[85]	validation_0-rmse:9.7458	validation_1-rmse:10.9063
[86]	validation_0-rmse:9.74557	validation_1-rmse:10.9064
[87]	validation_0-rmse:9.73543	validation_1-rmse:10.9055
[88]	validation_0-rmse:9.7333	validation_1-rmse:10.9057
[89]	validation_0-rmse:9.72811	validation_1-rmse:10.9077
[90]	validation_0-rmse:9.72292	validation_1-rmse:10.906
[91]	validation_0-rmse:9.71179	validation_1-rmse:10.9054
[92]	validation_0-rmse:9.71026	validation_1-rmse:10.905
[93]	validation_0-rmse:9.7047	validation_1-rmse:10.906
[94]	validation_0-rmse:9.70096	validation_1-rmse:10.9086
[95]	validation_0-rmse:9.69157	validation_1-rmse:10.904
[96]	validation_0-rmse:9.68971	validation_1-rmse:10.9044
[97]	validation_0-rmse:9.68638	validation_1-rmse:10.9019
[98]	validation_0-rmse:9.68126	validation_1-rmse:10.9002
[99]	validation_0-rmse:9.67794	validation_1-rmse:10.8998
[100]	validation_0-rmse:9.67161	validation_1-rmse:10.8974
[101]	validation_0-rmse:9.66816	validation_1-rmse:10.8969
[102]	validation_0-rmse:9.6626	validation_1-rmse:10.8972
[103]	validation_0-rmse:9.65862	validation_1-rmse:10.8952
[104]	validation_0-rmse:9.65168	validation_1-rmse:10.892
[105]	validation_0-rmse:9.64986	validation_1-rmse:10.8918
[106]	validation_0-rmse:9.64445	validation_1-rmse:10.8927
[107]	validation_0-rmse:9.63481	validation_1-rmse:10.8884
[108]	validation_0-rmse:9.63272	validation_1-rmse:10.8901
[109]	validation_0-rmse:9.6305	validation_1-rmse:10.8927
[110]	validation_0-rmse:9.62778	validation_1-rmse:10.8913
[111]	validation_0-rmse:9.62622	validation_1-rmse:10.8904
[112]	validation_0-rmse:9.6177	validation_1-rmse:10.8913
[113]	validation_0-rmse:9.61399	validation_1-rmse:10.892
[114]	validation_0-rmse:9.6082	validation_1-rmse:10.8937
[115]	validation_0-rmse:9.60541	validation_1-rmse:10.8944
[116]	validation_0-rmse:9.599	validation_1-rmse:10.8914
[117]	validation_0-rmse:9.59538	validation_1-rmse:10.8898
[118]	validation_0-rmse:9.58798	validation_1-rmse:10.8877
[119]	validation_0-rmse:9.5849	validation_1-rmse:10.8868
[120]	validation_0-rmse:9.57823	validation_1-rmse:10.8854
[121]	validation_0-rmse:9.5737	validation_1-rmse:10.8844
[122]	validation_0-rmse:9.57102	validation_1-rmse:10.8879
[123]	validation_0-rmse:9.56225	validation_1-rmse:10.8858
[124]	validation_0-rmse:9.55732	validation_1-rmse:10.8856
[125]	validation_0-rmse:9.55383	validation_1-rmse:10.8854
[126]	validation_0-rmse:9.55045	validation_1-rmse:10.887
[127]	validation_0-rmse:9.53874	validation_1-rmse:10.8839
[128]	validation_0-rmse:9.52769	validation_1-rmse:10.8838
[129]	validation_0-rmse:9.52098	validation_1-rmse:10.8859
[130]	validation_0-rmse:9.51835	validation_1-rmse:10.8851
[131]	validation_0-rmse:9.51498	validation_1-rmse:10.8833
[132]	validation_0-rmse:9.51182	validation_1-rmse:10.8868
[133]	validation_0-rmse:9.50615	validation_1-rmse:10.8841
[134]	validation_0-rmse:9.50012	validation_1-rmse:10.8825
[135]	validation_0-rmse:9.49999	validation_1-rmse:10.8825
[136]	validation_0-rmse:9.49681	validation_1-rmse:10.8821
[137]	validation_0-rmse:9.49381	validation_1-rmse:10.8843
[138]	validation_0-rmse:9.48839	validation_1-rmse:10.8892
[139]	validation_0-rmse:9.48626	validation_1-rmse:10.8902
[140]	validation_0-rmse:9.47929	validation_1-rmse:10.8916
[141]	validation_0-rmse:9.47126	validation_1-rmse:10.8905
[142]	validation_0-rmse:9.4651	validation_1-rmse:10.8889
[143]	validation_0-rmse:9.45986	validation_1-rmse:10.8954
[144]	validation_0-rmse:9.45676	validation_1-rmse:10.8979
[145]	validation_0-rmse:9.45239	validation_1-rmse:10.9024
[146]	validation_0-rmse:9.44764	validation_1-rmse:10.9003
[147]	validation_0-rmse:9.44553	validation_1-rmse:10.9008
[148]	validation_0-rmse:9.44168	validation_1-rmse:10.8984
[149]	validation_0-rmse:9.44052	validation_1-rmse:10.8959
[150]	validation_0-rmse:9.43703	validation_1-rmse:10.8979
[151]	validation_0-rmse:9.43508	validation_1-rmse:10.898
[152]	validation_0-rmse:9.43013	validation_1-rmse:10.8958
[153]	validation_0-rmse:9.42587	validation_1-rmse:10.8946
[154]	validation_0-rmse:9.42445	validation_1-rmse:10.8932
[155]	validation_0-rmse:9.42181	validation_1-rmse:10.8909
[156]	validation_0-rmse:9.4153	validation_1-rmse:10.8887
[157]	validation_0-rmse:9.41294	validation_1-rmse:10.8884
[158]	validation_0-rmse:9.40012	validation_1-rmse:10.8832
[159]	validation_0-rmse:9.39642	validation_1-rmse:10.8826
[160]	validation_0-rmse:9.39312	validation_1-rmse:10.8843
[161]	validation_0-rmse:9.38592	validation_1-rmse:10.8828
[162]	validation_0-rmse:9.38007	validation_1-rmse:10.8802
[163]	validation_0-rmse:9.37774	validation_1-rmse:10.8796
[164]	validation_0-rmse:9.37233	validation_1-rmse:10.8791
[165]	validation_0-rmse:9.36692	validation_1-rmse:10.8755
[166]	validation_0-rmse:9.36671	validation_1-rmse:10.8761
[167]	validation_0-rmse:9.36163	validation_1-rmse:10.8768
[168]	validation_0-rmse:9.3536	validation_1-rmse:10.8749
[169]	validation_0-rmse:9.35224	validation_1-rmse:10.8746
[170]	validation_0-rmse:9.35132	validation_1-rmse:10.8752
[171]	validation_0-rmse:9.34821	validation_1-rmse:10.8782
[172]	validation_0-rmse:9.34721	validation_1-rmse:10.8784
[173]	validation_0-rmse:9.34399	validation_1-rmse:10.8775
[174]	validation_0-rmse:9.33963	validation_1-rmse:10.8755
[175]	validation_0-rmse:9.33647	validation_1-rmse:10.8738
[176]	validation_0-rmse:9.33215	validation_1-rmse:10.8744
[177]	validation_0-rmse:9.32642	validation_1-rmse:10.8756
[178]	validation_0-rmse:9.32137	validation_1-rmse:10.8747
[179]	validation_0-rmse:9.31988	validation_1-rmse:10.8734
[180]	validation_0-rmse:9.31112	validation_1-rmse:10.8775
[181]	validation_0-rmse:9.30569	validation_1-rmse:10.8764
[182]	validation_0-rmse:9.30246	validation_1-rmse:10.8785
[183]	validation_0-rmse:9.29949	validation_1-rmse:10.8788
[184]	validation_0-rmse:9.29654	validation_1-rmse:10.8782
[185]	validation_0-rmse:9.29564	validation_1-rmse:10.8793
[186]	validation_0-rmse:9.28857	validation_1-rmse:10.8753
[187]	validation_0-rmse:9.28453	validation_1-rmse:10.8733
[188]	validation_0-rmse:9.28434	validation_1-rmse:10.8728
[189]	validation_0-rmse:9.28296	validation_1-rmse:10.8716
[190]	validation_0-rmse:9.28288	validation_1-rmse:10.8716
[191]	validation_0-rmse:9.28067	validation_1-rmse:10.8716
[192]	validation_0-rmse:9.27669	validation_1-rmse:10.8713
[193]	validation_0-rmse:9.27528	validation_1-rmse:10.8715
[194]	validation_0-rmse:9.27088	validation_1-rmse:10.8705
[195]	validation_0-rmse:9.26658	validation_1-rmse:10.8722
[196]	validation_0-rmse:9.26367	validation_1-rmse:10.8723
[197]	validation_0-rmse:9.26193	validation_1-rmse:10.8709
[198]	validation_0-rmse:9.26055	validation_1-rmse:10.8713
[199]	validation_0-rmse:9.26001	validation_1-rmse:10.8703
[200]	validation_0-rmse:9.25642	validation_1-rmse:10.8687
[201]	validation_0-rmse:9.25267	validation_1-rmse:10.8689
[202]	validation_0-rmse:9.25015	validation_1-rmse:10.8712
[203]	validation_0-rmse:9.247	validation_1-rmse:10.8733
[204]	validation_0-rmse:9.24692	validation_1-rmse:10.8733
[205]	validation_0-rmse:9.24243	validation_1-rmse:10.8711
[206]	validation_0-rmse:9.24076	validation_1-rmse:10.8709
[207]	validation_0-rmse:9.23795	validation_1-rmse:10.874
[208]	validation_0-rmse:9.23564	validation_1-rmse:10.8734
[209]	validation_0-rmse:9.2311	validation_1-rmse:10.8735
[210]	validation_0-rmse:9.22805	validation_1-rmse:10.8726
[211]	validation_0-rmse:9.22489	validation_1-rmse:10.8706
[212]	validation_0-rmse:9.22377	validation_1-rmse:10.8707
[213]	validation_0-rmse:9.22358	validation_1-rmse:10.8703
[214]	validation_0-rmse:9.22274	validation_1-rmse:10.8715
[215]	validation_0-rmse:9.22122	validation_1-rmse:10.8708
[216]	validation_0-rmse:9.21997	validation_1-rmse:10.8718
[217]	validation_0-rmse:9.2158	validation_1-rmse:10.8683
[218]	validation_0-rmse:9.21575	validation_1-rmse:10.8683
[219]	validation_0-rmse:9.21242	validation_1-rmse:10.8685
[220]	validation_0-rmse:9.21237	validation_1-rmse:10.8685
[221]	validation_0-rmse:9.21119	validation_1-rmse:10.8701
[222]	validation_0-rmse:9.20818	validation_1-rmse:10.8703
[223]	validation_0-rmse:9.20387	validation_1-rmse:10.8707
[224]	validation_0-rmse:9.20239	validation_1-rmse:10.8696
[225]	validation_0-rmse:9.2006	validation_1-rmse:10.8706
[226]	validation_0-rmse:9.20056	validation_1-rmse:10.8706
[227]	validation_0-rmse:9.19852	validation_1-rmse:10.8709
[228]	validation_0-rmse:9.19539	validation_1-rmse:10.8714
[229]	validation_0-rmse:9.19339	validation_1-rmse:10.8711
[230]	validation_0-rmse:9.18931	validation_1-rmse:10.8712
[231]	validation_0-rmse:9.18505	validation_1-rmse:10.8758
[232]	validation_0-rmse:9.1787	validation_1-rmse:10.8743
[233]	validation_0-rmse:9.17222	validation_1-rmse:10.871
[234]	validation_0-rmse:9.16878	validation_1-rmse:10.872
[235]	validation_0-rmse:9.16574	validation_1-rmse:10.8712
[236]	validation_0-rmse:9.16398	validation_1-rmse:10.8721
[237]	validation_0-rmse:9.15871	validation_1-rmse:10.8733
[238]	validation_0-rmse:9.15512	validation_1-rmse:10.8747
[239]	validation_0-rmse:9.15453	validation_1-rmse:10.8748
[240]	validation_0-rmse:9.14934	validation_1-rmse:10.877
[241]	validation_0-rmse:9.14805	validation_1-rmse:10.8774
[242]	validation_0-rmse:9.1446	validation_1-rmse:10.8781
[243]	validation_0-rmse:9.14235	validation_1-rmse:10.8775
[244]	validation_0-rmse:9.13899	validation_1-rmse:10.8782
[245]	validation_0-rmse:9.13542	validation_1-rmse:10.8803
[246]	validation_0-rmse:9.13093	validation_1-rmse:10.8784
[247]	validation_0-rmse:9.1309	validation_1-rmse:10.8784
[248]	validation_0-rmse:9.12694	validation_1-rmse:10.8753
[249]	validation_0-rmse:9.12529	validation_1-rmse:10.8747
[250]	validation_0-rmse:9.12526	validation_1-rmse:10.8747
[251]	validation_0-rmse:9.12422	validation_1-rmse:10.8748
[252]	validation_0-rmse:9.12382	validation_1-rmse:10.8753
[253]	validation_0-rmse:9.12199	validation_1-rmse:10.8749
[254]	validation_0-rmse:9.1208	validation_1-rmse:10.8748
[255]	validation_0-rmse:9.11601	validation_1-rmse:10.8721
[256]	validation_0-rmse:9.11041	validation_1-rmse:10.8691
[257]	validation_0-rmse:9.10451	validation_1-rmse:10.8668
[258]	validation_0-rmse:9.10258	validation_1-rmse:10.8665
[259]	validation_0-rmse:9.09959	validation_1-rmse:10.8686
[260]	validation_0-rmse:9.09757	validation_1-rmse:10.8682
[261]	validation_0-rmse:9.09655	validation_1-rmse:10.868
[262]	validation_0-rmse:9.09469	validation_1-rmse:10.8684
[263]	validation_0-rmse:9.09036	validation_1-rmse:10.8645
[264]	validation_0-rmse:9.0889	validation_1-rmse:10.8677
[265]	validation_0-rmse:9.08714	validation_1-rmse:10.8685
[266]	validation_0-rmse:9.084	validation_1-rmse:10.8684
[267]	validation_0-rmse:9.08398	validation_1-rmse:10.8684
[268]	validation_0-rmse:9.07729	validation_1-rmse:10.8652
[269]	validation_0-rmse:9.07533	validation_1-rmse:10.8665
[270]	validation_0-rmse:9.07363	validation_1-rmse:10.8697
[271]	validation_0-rmse:9.06567	validation_1-rmse:10.8686
[272]	validation_0-rmse:9.06526	validation_1-rmse:10.8682
[273]	validation_0-rmse:9.06525	validation_1-rmse:10.8682
[274]	validation_0-rmse:9.06201	validation_1-rmse:10.8681
[275]	validation_0-rmse:9.06059	validation_1-rmse:10.8684
[276]	validation_0-rmse:9.05896	validation_1-rmse:10.8685
[277]	validation_0-rmse:9.05645	validation_1-rmse:10.8685
[278]	validation_0-rmse:9.05457	validation_1-rmse:10.8685
[279]	validation_0-rmse:9.05366	validation_1-rmse:10.8684
[280]	validation_0-rmse:9.04994	validation_1-rmse:10.8678
[281]	validation_0-rmse:9.04992	validation_1-rmse:10.8679
[282]	validation_0-rmse:9.04646	validation_1-rmse:10.8668
[283]	validation_0-rmse:9.04644	validation_1-rmse:10.8668
[284]	validation_0-rmse:9.04498	validation_1-rmse:10.8672
[285]	validation_0-rmse:9.0446	validation_1-rmse:10.8672
[286]	validation_0-rmse:9.04232	validation_1-rmse:10.8667
[287]	validation_0-rmse:9.04074	validation_1-rmse:10.866
[288]	validation_0-rmse:9.03816	validation_1-rmse:10.8659
[289]	validation_0-rmse:9.03814	validation_1-rmse:10.866
[290]	validation_0-rmse:9.03813	validation_1-rmse:10.866
[291]	validation_0-rmse:9.03555	validation_1-rmse:10.8672
[292]	validation_0-rmse:9.03569	validation_1-rmse:10.8669
[293]	validation_0-rmse:9.03388	validation_1-rmse:10.8695
[294]	validation_0-rmse:9.03386	validation_1-rmse:10.8695
[295]	validation_0-rmse:9.02877	validation_1-rmse:10.8682
[296]	validation_0-rmse:9.02848	validation_1-rmse:10.8664
[297]	validation_0-rmse:9.02764	validation_1-rmse:10.8669
[298]	validation_0-rmse:9.02635	validation_1-rmse:10.8671
[299]	validation_0-rmse:9.02531	validation_1-rmse:10.8671
[300]	validation_0-rmse:9.02355	validation_1-rmse:10.8677
[301]	validation_0-rmse:9.02176	validation_1-rmse:10.867
[302]	validation_0-rmse:9.02001	validation_1-rmse:10.8679
[303]	validation_0-rmse:9.01877	validation_1-rmse:10.8674
[304]	validation_0-rmse:9.01875	validation_1-rmse:10.8674
[305]	validation_0-rmse:9.01874	validation_1-rmse:10.8674
[306]	validation_0-rmse:9.01425	validation_1-rmse:10.8656
[307]	validation_0-rmse:9.01323	validation_1-rmse:10.8669
[308]	validation_0-rmse:9.01072	validation_1-rmse:10.8659
[309]	validation_0-rmse:9.01071	validation_1-rmse:10.8659
[310]	validation_0-rmse:9.00897	validation_1-rmse:10.8675
[311]	validation_0-rmse:9.00896	validation_1-rmse:10.8675
[312]	validation_0-rmse:9.0079	validation_1-rmse:10.8677
[313]	validation_0-rmse:9.00789	validation_1-rmse:10.8677
Stopping. Best iteration:
[263]	validation_0-rmse:9.09036	validation_1-rmse:10.8645

[0]	validation_0-rmse:24.2342	validation_1-rmse:24.5521
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3219	validation_1-rmse:22.6466
[2]	validation_0-rmse:20.6402	validation_1-rmse:20.9729
[3]	validation_0-rmse:19.1697	validation_1-rmse:19.5127
[4]	validation_0-rmse:17.8895	validation_1-rmse:18.2436
[5]	validation_0-rmse:16.7748	validation_1-rmse:17.1384
[6]	validation_0-rmse:15.8102	validation_1-rmse:16.1859
[7]	validation_0-rmse:14.9815	validation_1-rmse:15.367
[8]	validation_0-rmse:14.2767	validation_1-rmse:14.6737
[9]	validation_0-rmse:13.6739	validation_1-rmse:14.0789
[10]	validation_0-rmse:13.1599	validation_1-rmse:13.5762
[11]	validation_0-rmse:12.7302	validation_1-rmse:13.1571
[12]	validation_0-rmse:12.3666	validation_1-rmse:12.8065
[13]	validation_0-rmse:12.061	validation_1-rmse:12.5119
[14]	validation_0-rmse:11.8054	validation_1-rmse:12.2641
[15]	validation_0-rmse:11.5959	validation_1-rmse:12.0614
[16]	validation_0-rmse:11.4153	validation_1-rmse:11.8919
[17]	validation_0-rmse:11.2685	validation_1-rmse:11.751
[18]	validation_0-rmse:11.1448	validation_1-rmse:11.6318
[19]	validation_0-rmse:11.0462	validation_1-rmse:11.5365
[20]	validation_0-rmse:10.9596	validation_1-rmse:11.4563
[21]	validation_0-rmse:10.8912	validation_1-rmse:11.392
[22]	validation_0-rmse:10.8281	validation_1-rmse:11.3351
[23]	validation_0-rmse:10.7785	validation_1-rmse:11.2908
[24]	validation_0-rmse:10.7316	validation_1-rmse:11.248
[25]	validation_0-rmse:10.6901	validation_1-rmse:11.2056
[26]	validation_0-rmse:10.6597	validation_1-rmse:11.176
[27]	validation_0-rmse:10.6318	validation_1-rmse:11.1534
[28]	validation_0-rmse:10.6013	validation_1-rmse:11.1303
[29]	validation_0-rmse:10.5803	validation_1-rmse:11.114
[30]	validation_0-rmse:10.5627	validation_1-rmse:11.1018
[31]	validation_0-rmse:10.5368	validation_1-rmse:11.0944
[32]	validation_0-rmse:10.525	validation_1-rmse:11.0852
[33]	validation_0-rmse:10.5151	validation_1-rmse:11.0789
[34]	validation_0-rmse:10.4997	validation_1-rmse:11.0733
[35]	validation_0-rmse:10.4859	validation_1-rmse:11.0681
[36]	validation_0-rmse:10.4753	validation_1-rmse:11.06
[37]	validation_0-rmse:10.4686	validation_1-rmse:11.0539
[38]	validation_0-rmse:10.4622	validation_1-rmse:11.0503
[39]	validation_0-rmse:10.4513	validation_1-rmse:11.0443
[40]	validation_0-rmse:10.4446	validation_1-rmse:11.0392
[41]	validation_0-rmse:10.44	validation_1-rmse:11.0372
[42]	validation_0-rmse:10.4351	validation_1-rmse:11.036
[43]	validation_0-rmse:10.4285	validation_1-rmse:11.0309
[44]	validation_0-rmse:10.4199	validation_1-rmse:11.0295
[45]	validation_0-rmse:10.4113	validation_1-rmse:11.0288
[46]	validation_0-rmse:10.3979	validation_1-rmse:11.0253
[47]	validation_0-rmse:10.3936	validation_1-rmse:11.0227
[48]	validation_0-rmse:10.3858	validation_1-rmse:11.0204
[49]	validation_0-rmse:10.3793	validation_1-rmse:11.0164
[50]	validation_0-rmse:10.3716	validation_1-rmse:11.021
[51]	validation_0-rmse:10.3623	validation_1-rmse:11.0292
[52]	validation_0-rmse:10.3581	validation_1-rmse:11.0289
[53]	validation_0-rmse:10.3565	validation_1-rmse:11.0272
[54]	validation_0-rmse:10.3487	validation_1-rmse:11.0241
[55]	validation_0-rmse:10.3391	validation_1-rmse:11.0272
[56]	validation_0-rmse:10.3375	validation_1-rmse:11.033
[57]	validation_0-rmse:10.3294	validation_1-rmse:11.033
[58]	validation_0-rmse:10.3246	validation_1-rmse:11.0289
[59]	validation_0-rmse:10.3159	validation_1-rmse:11.0261
[60]	validation_0-rmse:10.3127	validation_1-rmse:11.0263
[61]	validation_0-rmse:10.3089	validation_1-rmse:11.0315
[62]	validation_0-rmse:10.3056	validation_1-rmse:11.0279
[63]	validation_0-rmse:10.2994	validation_1-rmse:11.0275
[64]	validation_0-rmse:10.2951	validation_1-rmse:11.0252
[65]	validation_0-rmse:10.2925	validation_1-rmse:11.0246
[66]	validation_0-rmse:10.2902	validation_1-rmse:11.0244
[67]	validation_0-rmse:10.2854	validation_1-rmse:11.0206
[68]	validation_0-rmse:10.2818	validation_1-rmse:11.016
[69]	validation_0-rmse:10.2747	validation_1-rmse:11.0125
[70]	validation_0-rmse:10.2619	validation_1-rmse:11.0142
[71]	validation_0-rmse:10.2592	validation_1-rmse:11.0074
[72]	validation_0-rmse:10.2541	validation_1-rmse:11.0205
[73]	validation_0-rmse:10.2481	validation_1-rmse:11.0287
[74]	validation_0-rmse:10.2462	validation_1-rmse:11.0317
[75]	validation_0-rmse:10.2416	validation_1-rmse:11.0313
[76]	validation_0-rmse:10.239	validation_1-rmse:11.0305
[77]	validation_0-rmse:10.2328	validation_1-rmse:11.0341
[78]	validation_0-rmse:10.2269	validation_1-rmse:11.0337
[79]	validation_0-rmse:10.2249	validation_1-rmse:11.0328
[80]	validation_0-rmse:10.2223	validation_1-rmse:11.0303
[81]	validation_0-rmse:10.2152	validation_1-rmse:11.0303
[82]	validation_0-rmse:10.2124	validation_1-rmse:11.0372
[83]	validation_0-rmse:10.2081	validation_1-rmse:11.0384
[84]	validation_0-rmse:10.203	validation_1-rmse:11.0382
[85]	validation_0-rmse:10.1983	validation_1-rmse:11.0466
[86]	validation_0-rmse:10.1964	validation_1-rmse:11.0462
[87]	validation_0-rmse:10.1957	validation_1-rmse:11.0453
[88]	validation_0-rmse:10.1922	validation_1-rmse:11.0445
[89]	validation_0-rmse:10.1891	validation_1-rmse:11.0429
[90]	validation_0-rmse:10.1841	validation_1-rmse:11.0421
[91]	validation_0-rmse:10.1774	validation_1-rmse:11.0395
[92]	validation_0-rmse:10.1747	validation_1-rmse:11.0375
[93]	validation_0-rmse:10.1706	validation_1-rmse:11.0356
[94]	validation_0-rmse:10.1674	validation_1-rmse:11.0417
[95]	validation_0-rmse:10.1655	validation_1-rmse:11.0417
[96]	validation_0-rmse:10.1645	validation_1-rmse:11.0418
[97]	validation_0-rmse:10.1635	validation_1-rmse:11.0494
[98]	validation_0-rmse:10.1603	validation_1-rmse:11.0507
[99]	validation_0-rmse:10.1569	validation_1-rmse:11.0497
[100]	validation_0-rmse:10.1548	validation_1-rmse:11.0502
[101]	validation_0-rmse:10.1514	validation_1-rmse:11.0597
[102]	validation_0-rmse:10.148	validation_1-rmse:11.0589
[103]	validation_0-rmse:10.1428	validation_1-rmse:11.056
[104]	validation_0-rmse:10.1397	validation_1-rmse:11.0563
[105]	validation_0-rmse:10.1369	validation_1-rmse:11.0566
[106]	validation_0-rmse:10.1346	validation_1-rmse:11.0645
[107]	validation_0-rmse:10.1305	validation_1-rmse:11.0675
[108]	validation_0-rmse:10.1289	validation_1-rmse:11.0679
[109]	validation_0-rmse:10.1275	validation_1-rmse:11.0669
[110]	validation_0-rmse:10.1256	validation_1-rmse:11.0661
[111]	validation_0-rmse:10.1256	validation_1-rmse:11.0672
[112]	validation_0-rmse:10.1202	validation_1-rmse:11.0657
[113]	validation_0-rmse:10.1182	validation_1-rmse:11.0692
[114]	validation_0-rmse:10.1127	validation_1-rmse:11.064
[115]	validation_0-rmse:10.1107	validation_1-rmse:11.0635
[116]	validation_0-rmse:10.1084	validation_1-rmse:11.0648
[117]	validation_0-rmse:10.102	validation_1-rmse:11.0634
[118]	validation_0-rmse:10.0998	validation_1-rmse:11.0595
[119]	validation_0-rmse:10.0991	validation_1-rmse:11.0592
[120]	validation_0-rmse:10.097	validation_1-rmse:11.0588
[121]	validation_0-rmse:10.0935	validation_1-rmse:11.0567
Stopping. Best iteration:
[71]	validation_0-rmse:10.2592	validation_1-rmse:11.0074

[0]	validation_0-rmse:24.4672	validation_1-rmse:24.7822
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.5315	validation_1-rmse:22.8525
[2]	validation_0-rmse:20.8299	validation_1-rmse:21.1573
[3]	validation_0-rmse:19.469	validation_1-rmse:19.8074
[4]	validation_0-rmse:18.4902	validation_1-rmse:18.8425
[5]	validation_0-rmse:17.4739	validation_1-rmse:17.8342
[6]	validation_0-rmse:16.4259	validation_1-rmse:16.7954
[7]	validation_0-rmse:15.6768	validation_1-rmse:16.0535
[8]	validation_0-rmse:14.8937	validation_1-rmse:15.2782
[9]	validation_0-rmse:14.2195	validation_1-rmse:14.6137
[10]	validation_0-rmse:13.6425	validation_1-rmse:14.0428
[11]	validation_0-rmse:13.1564	validation_1-rmse:13.5632
[12]	validation_0-rmse:12.8385	validation_1-rmse:13.2524
[13]	validation_0-rmse:12.6084	validation_1-rmse:13.0283
[14]	validation_0-rmse:12.3376	validation_1-rmse:12.7627
[15]	validation_0-rmse:12.0673	validation_1-rmse:12.496
[16]	validation_0-rmse:11.8811	validation_1-rmse:12.3123
[17]	validation_0-rmse:11.6836	validation_1-rmse:12.119
[18]	validation_0-rmse:11.5516	validation_1-rmse:11.989
[19]	validation_0-rmse:11.4092	validation_1-rmse:11.8495
[20]	validation_0-rmse:11.3312	validation_1-rmse:11.7753
[21]	validation_0-rmse:11.2536	validation_1-rmse:11.6994
[22]	validation_0-rmse:11.2076	validation_1-rmse:11.6564
[23]	validation_0-rmse:11.1226	validation_1-rmse:11.5733
[24]	validation_0-rmse:11.0816	validation_1-rmse:11.5336
[25]	validation_0-rmse:11.0152	validation_1-rmse:11.4689
[26]	validation_0-rmse:10.9876	validation_1-rmse:11.4412
[27]	validation_0-rmse:10.9416	validation_1-rmse:11.3957
[28]	validation_0-rmse:10.9112	validation_1-rmse:11.3685
[29]	validation_0-rmse:10.8936	validation_1-rmse:11.3519
[30]	validation_0-rmse:10.8572	validation_1-rmse:11.318
[31]	validation_0-rmse:10.8251	validation_1-rmse:11.2857
[32]	validation_0-rmse:10.8135	validation_1-rmse:11.2754
[33]	validation_0-rmse:10.7867	validation_1-rmse:11.2521
[34]	validation_0-rmse:10.7683	validation_1-rmse:11.2339
[35]	validation_0-rmse:10.7607	validation_1-rmse:11.228
[36]	validation_0-rmse:10.7442	validation_1-rmse:11.2135
[37]	validation_0-rmse:10.7305	validation_1-rmse:11.1999
[38]	validation_0-rmse:10.7157	validation_1-rmse:11.186
[39]	validation_0-rmse:10.7043	validation_1-rmse:11.1756
[40]	validation_0-rmse:10.693	validation_1-rmse:11.1665
[41]	validation_0-rmse:10.6816	validation_1-rmse:11.1572
[42]	validation_0-rmse:10.6745	validation_1-rmse:11.1505
[43]	validation_0-rmse:10.667	validation_1-rmse:11.1447
[44]	validation_0-rmse:10.659	validation_1-rmse:11.1439
[45]	validation_0-rmse:10.6502	validation_1-rmse:11.1401
[46]	validation_0-rmse:10.6432	validation_1-rmse:11.1343
[47]	validation_0-rmse:10.6323	validation_1-rmse:11.1249
[48]	validation_0-rmse:10.6207	validation_1-rmse:11.1156
[49]	validation_0-rmse:10.6126	validation_1-rmse:11.1126
[50]	validation_0-rmse:10.6066	validation_1-rmse:11.1108
[51]	validation_0-rmse:10.5997	validation_1-rmse:11.1053
[52]	validation_0-rmse:10.5931	validation_1-rmse:11.101
[53]	validation_0-rmse:10.5854	validation_1-rmse:11.0934
[54]	validation_0-rmse:10.5763	validation_1-rmse:11.0912
[55]	validation_0-rmse:10.5709	validation_1-rmse:11.0864
[56]	validation_0-rmse:10.5647	validation_1-rmse:11.0813
[57]	validation_0-rmse:10.5606	validation_1-rmse:11.0787
[58]	validation_0-rmse:10.5564	validation_1-rmse:11.0753
[59]	validation_0-rmse:10.5535	validation_1-rmse:11.0723
[60]	validation_0-rmse:10.548	validation_1-rmse:11.0697
[61]	validation_0-rmse:10.5435	validation_1-rmse:11.0676
[62]	validation_0-rmse:10.5409	validation_1-rmse:11.0662
[63]	validation_0-rmse:10.5374	validation_1-rmse:11.0631
[64]	validation_0-rmse:10.5351	validation_1-rmse:11.062
[65]	validation_0-rmse:10.5327	validation_1-rmse:11.0608
[66]	validation_0-rmse:10.5256	validation_1-rmse:11.0599
[67]	validation_0-rmse:10.5206	validation_1-rmse:11.0565
[68]	validation_0-rmse:10.5156	validation_1-rmse:11.0541
[69]	validation_0-rmse:10.5114	validation_1-rmse:11.0518
[70]	validation_0-rmse:10.5059	validation_1-rmse:11.0504
[71]	validation_0-rmse:10.5022	validation_1-rmse:11.0478
[72]	validation_0-rmse:10.4932	validation_1-rmse:11.0473
[73]	validation_0-rmse:10.4902	validation_1-rmse:11.0464
[74]	validation_0-rmse:10.486	validation_1-rmse:11.0432
[75]	validation_0-rmse:10.4801	validation_1-rmse:11.0457
[76]	validation_0-rmse:10.4735	validation_1-rmse:11.0431
[77]	validation_0-rmse:10.4707	validation_1-rmse:11.0413
[78]	validation_0-rmse:10.4655	validation_1-rmse:11.0378
[79]	validation_0-rmse:10.4625	validation_1-rmse:11.0356
[80]	validation_0-rmse:10.4595	validation_1-rmse:11.0352
[81]	validation_0-rmse:10.4566	validation_1-rmse:11.0331
[82]	validation_0-rmse:10.4537	validation_1-rmse:11.0323
[83]	validation_0-rmse:10.4508	validation_1-rmse:11.0312
[84]	validation_0-rmse:10.4477	validation_1-rmse:11.0285
[85]	validation_0-rmse:10.4442	validation_1-rmse:11.0263
[86]	validation_0-rmse:10.4424	validation_1-rmse:11.0256
[87]	validation_0-rmse:10.4402	validation_1-rmse:11.0242
[88]	validation_0-rmse:10.439	validation_1-rmse:11.0231
[89]	validation_0-rmse:10.4342	validation_1-rmse:11.0213
[90]	validation_0-rmse:10.4321	validation_1-rmse:11.0199
[91]	validation_0-rmse:10.4285	validation_1-rmse:11.0188
[92]	validation_0-rmse:10.4265	validation_1-rmse:11.0174
[93]	validation_0-rmse:10.4243	validation_1-rmse:11.0178
[94]	validation_0-rmse:10.4227	validation_1-rmse:11.0166
[95]	validation_0-rmse:10.4206	validation_1-rmse:11.0144
[96]	validation_0-rmse:10.4191	validation_1-rmse:11.014
[97]	validation_0-rmse:10.4149	validation_1-rmse:11.01
[98]	validation_0-rmse:10.4114	validation_1-rmse:11.0066
[99]	validation_0-rmse:10.4103	validation_1-rmse:11.0064
[100]	validation_0-rmse:10.4085	validation_1-rmse:11.0078
[101]	validation_0-rmse:10.4076	validation_1-rmse:11.008
[102]	validation_0-rmse:10.4062	validation_1-rmse:11.0071
[103]	validation_0-rmse:10.4049	validation_1-rmse:11.0063
[104]	validation_0-rmse:10.4002	validation_1-rmse:11.0035
[105]	validation_0-rmse:10.3991	validation_1-rmse:11.0031
[106]	validation_0-rmse:10.3969	validation_1-rmse:11.0022
[107]	validation_0-rmse:10.3941	validation_1-rmse:10.9999
[108]	validation_0-rmse:10.3927	validation_1-rmse:10.9999
[109]	validation_0-rmse:10.3913	validation_1-rmse:10.9983
[110]	validation_0-rmse:10.3878	validation_1-rmse:10.996
[111]	validation_0-rmse:10.3857	validation_1-rmse:10.9951
[112]	validation_0-rmse:10.384	validation_1-rmse:10.9949
[113]	validation_0-rmse:10.3805	validation_1-rmse:10.9932
[114]	validation_0-rmse:10.3772	validation_1-rmse:10.9911
[115]	validation_0-rmse:10.3722	validation_1-rmse:10.987
[116]	validation_0-rmse:10.3685	validation_1-rmse:10.985
[117]	validation_0-rmse:10.367	validation_1-rmse:10.984
[118]	validation_0-rmse:10.3644	validation_1-rmse:10.9833
[119]	validation_0-rmse:10.3627	validation_1-rmse:10.9817
[120]	validation_0-rmse:10.3615	validation_1-rmse:10.9823
[121]	validation_0-rmse:10.3603	validation_1-rmse:10.9813
[122]	validation_0-rmse:10.357	validation_1-rmse:10.979
[123]	validation_0-rmse:10.3551	validation_1-rmse:10.9777
[124]	validation_0-rmse:10.3503	validation_1-rmse:10.9746
[125]	validation_0-rmse:10.3493	validation_1-rmse:10.9735
[126]	validation_0-rmse:10.3483	validation_1-rmse:10.9736
[127]	validation_0-rmse:10.3449	validation_1-rmse:10.9734
[128]	validation_0-rmse:10.3421	validation_1-rmse:10.9723
[129]	validation_0-rmse:10.3395	validation_1-rmse:10.9713
[130]	validation_0-rmse:10.3384	validation_1-rmse:10.9703
[131]	validation_0-rmse:10.3349	validation_1-rmse:10.97
[132]	validation_0-rmse:10.3343	validation_1-rmse:10.9722
[133]	validation_0-rmse:10.3321	validation_1-rmse:10.9723
[134]	validation_0-rmse:10.329	validation_1-rmse:10.9765
[135]	validation_0-rmse:10.3284	validation_1-rmse:10.9761
[136]	validation_0-rmse:10.3274	validation_1-rmse:10.9754
[137]	validation_0-rmse:10.3261	validation_1-rmse:10.9745
[138]	validation_0-rmse:10.3233	validation_1-rmse:10.9732
[139]	validation_0-rmse:10.3221	validation_1-rmse:10.9731
[140]	validation_0-rmse:10.3196	validation_1-rmse:10.9726
[141]	validation_0-rmse:10.318	validation_1-rmse:10.9722
[142]	validation_0-rmse:10.3092	validation_1-rmse:10.9746
[143]	validation_0-rmse:10.3082	validation_1-rmse:10.9742
[144]	validation_0-rmse:10.3056	validation_1-rmse:10.9753
[145]	validation_0-rmse:10.3052	validation_1-rmse:10.9759
[146]	validation_0-rmse:10.3004	validation_1-rmse:10.9806
[147]	validation_0-rmse:10.2978	validation_1-rmse:10.9798
[148]	validation_0-rmse:10.297	validation_1-rmse:10.9794
[149]	validation_0-rmse:10.296	validation_1-rmse:10.9785
[150]	validation_0-rmse:10.2934	validation_1-rmse:10.9824
[151]	validation_0-rmse:10.2925	validation_1-rmse:10.981
[152]	validation_0-rmse:10.2904	validation_1-rmse:10.981
[153]	validation_0-rmse:10.2887	validation_1-rmse:10.9807
[154]	validation_0-rmse:10.2862	validation_1-rmse:10.9804
[155]	validation_0-rmse:10.2805	validation_1-rmse:10.9824
[156]	validation_0-rmse:10.28	validation_1-rmse:10.9821
[157]	validation_0-rmse:10.2788	validation_1-rmse:10.9817
[158]	validation_0-rmse:10.2782	validation_1-rmse:10.9814
[159]	validation_0-rmse:10.276	validation_1-rmse:10.9799
[160]	validation_0-rmse:10.275	validation_1-rmse:10.9793
[161]	validation_0-rmse:10.2744	validation_1-rmse:10.9793
[162]	validation_0-rmse:10.2721	validation_1-rmse:10.978
[163]	validation_0-rmse:10.2672	validation_1-rmse:10.9803
[164]	validation_0-rmse:10.2652	validation_1-rmse:10.9821
[165]	validation_0-rmse:10.2629	validation_1-rmse:10.982
[166]	validation_0-rmse:10.2626	validation_1-rmse:10.9834
[167]	validation_0-rmse:10.2619	validation_1-rmse:10.9831
[168]	validation_0-rmse:10.2613	validation_1-rmse:10.9828
[169]	validation_0-rmse:10.261	validation_1-rmse:10.9821
[170]	validation_0-rmse:10.2605	validation_1-rmse:10.9822
[171]	validation_0-rmse:10.2553	validation_1-rmse:10.9869
[172]	validation_0-rmse:10.2547	validation_1-rmse:10.9876
[173]	validation_0-rmse:10.2515	validation_1-rmse:10.9903
[174]	validation_0-rmse:10.2509	validation_1-rmse:10.9903
[175]	validation_0-rmse:10.2473	validation_1-rmse:10.9911
[176]	validation_0-rmse:10.2451	validation_1-rmse:10.9895
[177]	validation_0-rmse:10.2441	validation_1-rmse:10.9891
[178]	validation_0-rmse:10.2373	validation_1-rmse:10.995
[179]	validation_0-rmse:10.2371	validation_1-rmse:10.9948
[180]	validation_0-rmse:10.2332	validation_1-rmse:10.9981
[181]	validation_0-rmse:10.2315	validation_1-rmse:10.9968
Stopping. Best iteration:
[131]	validation_0-rmse:10.3349	validation_1-rmse:10.97

[0]	validation_0-rmse:24.2211	validation_1-rmse:24.5376
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3009	validation_1-rmse:22.6299
[2]	validation_0-rmse:20.6266	validation_1-rmse:20.9661
[3]	validation_0-rmse:19.1449	validation_1-rmse:19.4971
[4]	validation_0-rmse:17.8557	validation_1-rmse:18.2229
[5]	validation_0-rmse:16.7341	validation_1-rmse:17.1187
[6]	validation_0-rmse:15.7559	validation_1-rmse:16.1598
[7]	validation_0-rmse:14.923	validation_1-rmse:15.3504
[8]	validation_0-rmse:14.1996	validation_1-rmse:14.6424
[9]	validation_0-rmse:13.6243	validation_1-rmse:14.0813
[10]	validation_0-rmse:13.0919	validation_1-rmse:13.5625
[11]	validation_0-rmse:12.64	validation_1-rmse:13.1297
[12]	validation_0-rmse:12.2709	validation_1-rmse:12.7803
[13]	validation_0-rmse:11.9477	validation_1-rmse:12.4734
[14]	validation_0-rmse:11.6719	validation_1-rmse:12.2175
[15]	validation_0-rmse:11.4379	validation_1-rmse:12.0033
[16]	validation_0-rmse:11.246	validation_1-rmse:11.824
[17]	validation_0-rmse:11.0798	validation_1-rmse:11.6735
[18]	validation_0-rmse:10.9424	validation_1-rmse:11.5507
[19]	validation_0-rmse:10.8301	validation_1-rmse:11.4506
[20]	validation_0-rmse:10.7326	validation_1-rmse:11.3653
[21]	validation_0-rmse:10.6539	validation_1-rmse:11.2955
[22]	validation_0-rmse:10.5809	validation_1-rmse:11.234
[23]	validation_0-rmse:10.52	validation_1-rmse:11.181
[24]	validation_0-rmse:10.4679	validation_1-rmse:11.1413
[25]	validation_0-rmse:10.4167	validation_1-rmse:11.1053
[26]	validation_0-rmse:10.3793	validation_1-rmse:11.0768
[27]	validation_0-rmse:10.3469	validation_1-rmse:11.0505
[28]	validation_0-rmse:10.3182	validation_1-rmse:11.0315
[29]	validation_0-rmse:10.2836	validation_1-rmse:11.014
[30]	validation_0-rmse:10.2631	validation_1-rmse:10.998
[31]	validation_0-rmse:10.2376	validation_1-rmse:10.9819
[32]	validation_0-rmse:10.2102	validation_1-rmse:10.9706
[33]	validation_0-rmse:10.1929	validation_1-rmse:10.9613
[34]	validation_0-rmse:10.1776	validation_1-rmse:10.9515
[35]	validation_0-rmse:10.1623	validation_1-rmse:10.9441
[36]	validation_0-rmse:10.1506	validation_1-rmse:10.9379
[37]	validation_0-rmse:10.14	validation_1-rmse:10.9318
[38]	validation_0-rmse:10.1306	validation_1-rmse:10.9265
[39]	validation_0-rmse:10.1218	validation_1-rmse:10.9205
[40]	validation_0-rmse:10.1	validation_1-rmse:10.9107
[41]	validation_0-rmse:10.0843	validation_1-rmse:10.9039
[42]	validation_0-rmse:10.0751	validation_1-rmse:10.8991
[43]	validation_0-rmse:10.0647	validation_1-rmse:10.8952
[44]	validation_0-rmse:10.0586	validation_1-rmse:10.8924
[45]	validation_0-rmse:10.0436	validation_1-rmse:10.8856
[46]	validation_0-rmse:10.0262	validation_1-rmse:10.8808
[47]	validation_0-rmse:10.0155	validation_1-rmse:10.8751
[48]	validation_0-rmse:10.0005	validation_1-rmse:10.8699
[49]	validation_0-rmse:9.98933	validation_1-rmse:10.8699
[50]	validation_0-rmse:9.98025	validation_1-rmse:10.8657
[51]	validation_0-rmse:9.97033	validation_1-rmse:10.8632
[52]	validation_0-rmse:9.95657	validation_1-rmse:10.8616
[53]	validation_0-rmse:9.9494	validation_1-rmse:10.8591
[54]	validation_0-rmse:9.94206	validation_1-rmse:10.8537
[55]	validation_0-rmse:9.93726	validation_1-rmse:10.8518
[56]	validation_0-rmse:9.92895	validation_1-rmse:10.852
[57]	validation_0-rmse:9.91756	validation_1-rmse:10.8458
[58]	validation_0-rmse:9.91343	validation_1-rmse:10.8434
[59]	validation_0-rmse:9.90937	validation_1-rmse:10.8423
[60]	validation_0-rmse:9.89818	validation_1-rmse:10.8403
[61]	validation_0-rmse:9.89373	validation_1-rmse:10.8408
[62]	validation_0-rmse:9.88734	validation_1-rmse:10.8404
[63]	validation_0-rmse:9.87655	validation_1-rmse:10.8416
[64]	validation_0-rmse:9.87145	validation_1-rmse:10.8414
[65]	validation_0-rmse:9.86681	validation_1-rmse:10.8405
[66]	validation_0-rmse:9.85794	validation_1-rmse:10.8397
[67]	validation_0-rmse:9.83918	validation_1-rmse:10.8304
[68]	validation_0-rmse:9.83293	validation_1-rmse:10.8274
[69]	validation_0-rmse:9.82869	validation_1-rmse:10.8266
[70]	validation_0-rmse:9.81222	validation_1-rmse:10.8194
[71]	validation_0-rmse:9.80939	validation_1-rmse:10.8185
[72]	validation_0-rmse:9.79945	validation_1-rmse:10.8159
[73]	validation_0-rmse:9.79003	validation_1-rmse:10.8185
[74]	validation_0-rmse:9.78407	validation_1-rmse:10.8164
[75]	validation_0-rmse:9.78081	validation_1-rmse:10.8159
[76]	validation_0-rmse:9.77291	validation_1-rmse:10.8167
[77]	validation_0-rmse:9.7626	validation_1-rmse:10.8138
[78]	validation_0-rmse:9.75414	validation_1-rmse:10.8102
[79]	validation_0-rmse:9.75391	validation_1-rmse:10.81
[80]	validation_0-rmse:9.7496	validation_1-rmse:10.8089
[81]	validation_0-rmse:9.7494	validation_1-rmse:10.8089
[82]	validation_0-rmse:9.74685	validation_1-rmse:10.8088
[83]	validation_0-rmse:9.74665	validation_1-rmse:10.8088
[84]	validation_0-rmse:9.74648	validation_1-rmse:10.8086
[85]	validation_0-rmse:9.74538	validation_1-rmse:10.8086
[86]	validation_0-rmse:9.74311	validation_1-rmse:10.8083
[87]	validation_0-rmse:9.74296	validation_1-rmse:10.8082
[88]	validation_0-rmse:9.74282	validation_1-rmse:10.8081
[89]	validation_0-rmse:9.73518	validation_1-rmse:10.809
[90]	validation_0-rmse:9.73243	validation_1-rmse:10.8088
[91]	validation_0-rmse:9.73232	validation_1-rmse:10.8088
[92]	validation_0-rmse:9.73096	validation_1-rmse:10.8086
[93]	validation_0-rmse:9.73085	validation_1-rmse:10.8085
[94]	validation_0-rmse:9.73075	validation_1-rmse:10.8084
[95]	validation_0-rmse:9.73065	validation_1-rmse:10.8085
[96]	validation_0-rmse:9.72755	validation_1-rmse:10.8083
[97]	validation_0-rmse:9.72158	validation_1-rmse:10.8068
[98]	validation_0-rmse:9.72126	validation_1-rmse:10.8067
[99]	validation_0-rmse:9.72117	validation_1-rmse:10.8068
[100]	validation_0-rmse:9.7211	validation_1-rmse:10.8067
[101]	validation_0-rmse:9.72102	validation_1-rmse:10.8067
[102]	validation_0-rmse:9.72066	validation_1-rmse:10.8069
[103]	validation_0-rmse:9.72061	validation_1-rmse:10.807
[104]	validation_0-rmse:9.72054	validation_1-rmse:10.8069
[105]	validation_0-rmse:9.71146	validation_1-rmse:10.8058
[106]	validation_0-rmse:9.7114	validation_1-rmse:10.8058
[107]	validation_0-rmse:9.71134	validation_1-rmse:10.8059
[108]	validation_0-rmse:9.7113	validation_1-rmse:10.8059
[109]	validation_0-rmse:9.71126	validation_1-rmse:10.8059
[110]	validation_0-rmse:9.71122	validation_1-rmse:10.8059
[111]	validation_0-rmse:9.70969	validation_1-rmse:10.806
[112]	validation_0-rmse:9.70965	validation_1-rmse:10.806
[113]	validation_0-rmse:9.70962	validation_1-rmse:10.806
[114]	validation_0-rmse:9.70677	validation_1-rmse:10.8054
[115]	validation_0-rmse:9.70674	validation_1-rmse:10.8054
[116]	validation_0-rmse:9.70671	validation_1-rmse:10.8054
[117]	validation_0-rmse:9.70668	validation_1-rmse:10.8054
[118]	validation_0-rmse:9.70665	validation_1-rmse:10.8054
[119]	validation_0-rmse:9.70662	validation_1-rmse:10.8054
[120]	validation_0-rmse:9.7066	validation_1-rmse:10.8054
[121]	validation_0-rmse:9.70657	validation_1-rmse:10.8054
[122]	validation_0-rmse:9.70491	validation_1-rmse:10.8047
[123]	validation_0-rmse:9.70208	validation_1-rmse:10.8045
[124]	validation_0-rmse:9.70206	validation_1-rmse:10.8045
[125]	validation_0-rmse:9.70062	validation_1-rmse:10.8043
[126]	validation_0-rmse:9.7006	validation_1-rmse:10.8043
[127]	validation_0-rmse:9.70059	validation_1-rmse:10.8042
[128]	validation_0-rmse:9.69995	validation_1-rmse:10.8042
[129]	validation_0-rmse:9.69993	validation_1-rmse:10.8042
[130]	validation_0-rmse:9.69992	validation_1-rmse:10.8042
[131]	validation_0-rmse:9.69991	validation_1-rmse:10.8042
[132]	validation_0-rmse:9.6999	validation_1-rmse:10.8043
[133]	validation_0-rmse:9.69989	validation_1-rmse:10.8043
[134]	validation_0-rmse:9.69987	validation_1-rmse:10.8043
[135]	validation_0-rmse:9.69986	validation_1-rmse:10.8043
[136]	validation_0-rmse:9.69986	validation_1-rmse:10.8043
[137]	validation_0-rmse:9.69984	validation_1-rmse:10.8043
[138]	validation_0-rmse:9.69089	validation_1-rmse:10.8027
[139]	validation_0-rmse:9.69088	validation_1-rmse:10.8027
[140]	validation_0-rmse:9.68989	validation_1-rmse:10.8027
[141]	validation_0-rmse:9.68988	validation_1-rmse:10.8027
[142]	validation_0-rmse:9.68691	validation_1-rmse:10.805
[143]	validation_0-rmse:9.6869	validation_1-rmse:10.805
[144]	validation_0-rmse:9.68689	validation_1-rmse:10.8051
[145]	validation_0-rmse:9.68689	validation_1-rmse:10.8051
[146]	validation_0-rmse:9.68688	validation_1-rmse:10.8051
[147]	validation_0-rmse:9.68348	validation_1-rmse:10.8058
[148]	validation_0-rmse:9.68347	validation_1-rmse:10.8058
[149]	validation_0-rmse:9.68346	validation_1-rmse:10.8058
[150]	validation_0-rmse:9.68346	validation_1-rmse:10.8058
[151]	validation_0-rmse:9.68345	validation_1-rmse:10.8058
[152]	validation_0-rmse:9.68345	validation_1-rmse:10.8058
[153]	validation_0-rmse:9.68344	validation_1-rmse:10.8058
[154]	validation_0-rmse:9.68344	validation_1-rmse:10.8058
[155]	validation_0-rmse:9.68343	validation_1-rmse:10.8058
[156]	validation_0-rmse:9.68343	validation_1-rmse:10.8058
[157]	validation_0-rmse:9.68343	validation_1-rmse:10.8058
[158]	validation_0-rmse:9.67723	validation_1-rmse:10.8033
[159]	validation_0-rmse:9.67723	validation_1-rmse:10.8033
[160]	validation_0-rmse:9.67668	validation_1-rmse:10.8033
[161]	validation_0-rmse:9.67668	validation_1-rmse:10.8033
[162]	validation_0-rmse:9.67667	validation_1-rmse:10.8033
[163]	validation_0-rmse:9.66951	validation_1-rmse:10.8005
[164]	validation_0-rmse:9.66951	validation_1-rmse:10.8005
[165]	validation_0-rmse:9.66054	validation_1-rmse:10.797
[166]	validation_0-rmse:9.66054	validation_1-rmse:10.797
[167]	validation_0-rmse:9.65786	validation_1-rmse:10.797
[168]	validation_0-rmse:9.65786	validation_1-rmse:10.7969
[169]	validation_0-rmse:9.65785	validation_1-rmse:10.7969
[170]	validation_0-rmse:9.65785	validation_1-rmse:10.7969
[171]	validation_0-rmse:9.65785	validation_1-rmse:10.7969
[172]	validation_0-rmse:9.65785	validation_1-rmse:10.7969
[173]	validation_0-rmse:9.65784	validation_1-rmse:10.797
[174]	validation_0-rmse:9.65784	validation_1-rmse:10.797
[175]	validation_0-rmse:9.65784	validation_1-rmse:10.797
[176]	validation_0-rmse:9.65784	validation_1-rmse:10.797
[177]	validation_0-rmse:9.65783	validation_1-rmse:10.797
[178]	validation_0-rmse:9.65417	validation_1-rmse:10.7961
[179]	validation_0-rmse:9.65416	validation_1-rmse:10.7961
[180]	validation_0-rmse:9.65179	validation_1-rmse:10.7964
[181]	validation_0-rmse:9.65179	validation_1-rmse:10.7964
[182]	validation_0-rmse:9.65179	validation_1-rmse:10.7964
[183]	validation_0-rmse:9.65179	validation_1-rmse:10.7964
[184]	validation_0-rmse:9.65178	validation_1-rmse:10.7964
[185]	validation_0-rmse:9.65178	validation_1-rmse:10.7964
[186]	validation_0-rmse:9.65178	validation_1-rmse:10.7964
[187]	validation_0-rmse:9.64574	validation_1-rmse:10.7975
[188]	validation_0-rmse:9.64355	validation_1-rmse:10.7988
[189]	validation_0-rmse:9.64355	validation_1-rmse:10.7988
[190]	validation_0-rmse:9.64355	validation_1-rmse:10.7988
[191]	validation_0-rmse:9.64354	validation_1-rmse:10.7988
[192]	validation_0-rmse:9.64354	validation_1-rmse:10.7988
[193]	validation_0-rmse:9.63998	validation_1-rmse:10.7977
[194]	validation_0-rmse:9.63653	validation_1-rmse:10.7976
[195]	validation_0-rmse:9.63653	validation_1-rmse:10.7976
[196]	validation_0-rmse:9.63652	validation_1-rmse:10.7976
[197]	validation_0-rmse:9.63652	validation_1-rmse:10.7976
[198]	validation_0-rmse:9.63652	validation_1-rmse:10.7976
[199]	validation_0-rmse:9.63652	validation_1-rmse:10.7976
[200]	validation_0-rmse:9.63526	validation_1-rmse:10.7985
[201]	validation_0-rmse:9.63287	validation_1-rmse:10.7984
[202]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[203]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[204]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[205]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[206]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[207]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[208]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[209]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[210]	validation_0-rmse:9.62502	validation_1-rmse:10.7945
[211]	validation_0-rmse:9.61496	validation_1-rmse:10.7943
[212]	validation_0-rmse:9.61496	validation_1-rmse:10.7943
[213]	validation_0-rmse:9.60411	validation_1-rmse:10.7951
[214]	validation_0-rmse:9.6041	validation_1-rmse:10.7951
[215]	validation_0-rmse:9.60088	validation_1-rmse:10.795
[216]	validation_0-rmse:9.60087	validation_1-rmse:10.795
[217]	validation_0-rmse:9.60087	validation_1-rmse:10.795
[218]	validation_0-rmse:9.60087	validation_1-rmse:10.795
[219]	validation_0-rmse:9.60073	validation_1-rmse:10.7949
[220]	validation_0-rmse:9.60073	validation_1-rmse:10.7949
[221]	validation_0-rmse:9.59923	validation_1-rmse:10.795
[222]	validation_0-rmse:9.59923	validation_1-rmse:10.795
[223]	validation_0-rmse:9.59888	validation_1-rmse:10.7951
[224]	validation_0-rmse:9.59888	validation_1-rmse:10.7951
[225]	validation_0-rmse:9.59888	validation_1-rmse:10.7951
[226]	validation_0-rmse:9.59888	validation_1-rmse:10.7951
[227]	validation_0-rmse:9.59888	validation_1-rmse:10.7951
[228]	validation_0-rmse:9.59888	validation_1-rmse:10.7951
[229]	validation_0-rmse:9.59019	validation_1-rmse:10.7926
[230]	validation_0-rmse:9.59019	validation_1-rmse:10.7926
[231]	validation_0-rmse:9.59019	validation_1-rmse:10.7926
[232]	validation_0-rmse:9.57889	validation_1-rmse:10.7902
[233]	validation_0-rmse:9.57646	validation_1-rmse:10.7895
[234]	validation_0-rmse:9.57646	validation_1-rmse:10.7895
[235]	validation_0-rmse:9.57646	validation_1-rmse:10.7895
[236]	validation_0-rmse:9.5761	validation_1-rmse:10.7896
[237]	validation_0-rmse:9.5761	validation_1-rmse:10.7896
[238]	validation_0-rmse:9.57394	validation_1-rmse:10.7893
[239]	validation_0-rmse:9.57262	validation_1-rmse:10.7892
[240]	validation_0-rmse:9.57262	validation_1-rmse:10.7892
[241]	validation_0-rmse:9.57262	validation_1-rmse:10.7892
[242]	validation_0-rmse:9.57262	validation_1-rmse:10.7892
[243]	validation_0-rmse:9.57053	validation_1-rmse:10.7884
[244]	validation_0-rmse:9.57053	validation_1-rmse:10.7884
[245]	validation_0-rmse:9.57053	validation_1-rmse:10.7884
[246]	validation_0-rmse:9.57053	validation_1-rmse:10.7884
[247]	validation_0-rmse:9.56962	validation_1-rmse:10.7885
[248]	validation_0-rmse:9.56962	validation_1-rmse:10.7885
[249]	validation_0-rmse:9.56962	validation_1-rmse:10.7885
[250]	validation_0-rmse:9.56884	validation_1-rmse:10.7887
[251]	validation_0-rmse:9.56142	validation_1-rmse:10.7891
[252]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[253]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[254]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[255]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[256]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[257]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[258]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[259]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[260]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[261]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[262]	validation_0-rmse:9.55974	validation_1-rmse:10.7893
[263]	validation_0-rmse:9.55499	validation_1-rmse:10.7892
[264]	validation_0-rmse:9.55499	validation_1-rmse:10.7892
[265]	validation_0-rmse:9.55499	validation_1-rmse:10.7892
[266]	validation_0-rmse:9.55499	validation_1-rmse:10.7892
[267]	validation_0-rmse:9.54718	validation_1-rmse:10.7896
[268]	validation_0-rmse:9.54718	validation_1-rmse:10.7896
[269]	validation_0-rmse:9.54718	validation_1-rmse:10.7896
[270]	validation_0-rmse:9.54718	validation_1-rmse:10.7896
[271]	validation_0-rmse:9.54718	validation_1-rmse:10.7896
[272]	validation_0-rmse:9.5449	validation_1-rmse:10.7894
[273]	validation_0-rmse:9.5449	validation_1-rmse:10.7894
[274]	validation_0-rmse:9.54453	validation_1-rmse:10.7894
[275]	validation_0-rmse:9.54453	validation_1-rmse:10.7894
[276]	validation_0-rmse:9.54453	validation_1-rmse:10.7894
[277]	validation_0-rmse:9.54348	validation_1-rmse:10.7891
[278]	validation_0-rmse:9.54348	validation_1-rmse:10.7891
[279]	validation_0-rmse:9.54132	validation_1-rmse:10.7895
[280]	validation_0-rmse:9.54117	validation_1-rmse:10.7894
[281]	validation_0-rmse:9.54117	validation_1-rmse:10.7894
[282]	validation_0-rmse:9.54117	validation_1-rmse:10.7894
[283]	validation_0-rmse:9.54117	validation_1-rmse:10.7894
[284]	validation_0-rmse:9.54117	validation_1-rmse:10.7894
[285]	validation_0-rmse:9.53791	validation_1-rmse:10.7888
[286]	validation_0-rmse:9.53791	validation_1-rmse:10.7888
[287]	validation_0-rmse:9.53791	validation_1-rmse:10.7888
[288]	validation_0-rmse:9.53479	validation_1-rmse:10.7891
[289]	validation_0-rmse:9.53479	validation_1-rmse:10.7891
[290]	validation_0-rmse:9.53479	validation_1-rmse:10.7891
[291]	validation_0-rmse:9.53479	validation_1-rmse:10.7891
[292]	validation_0-rmse:9.53479	validation_1-rmse:10.7891
[293]	validation_0-rmse:9.53479	validation_1-rmse:10.7891
[294]	validation_0-rmse:9.53479	validation_1-rmse:10.7891
[295]	validation_0-rmse:9.53454	validation_1-rmse:10.7891
Stopping. Best iteration:
[245]	validation_0-rmse:9.57053	validation_1-rmse:10.7884

[0]	validation_0-rmse:24.2303	validation_1-rmse:24.547
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3134	validation_1-rmse:22.6385
[2]	validation_0-rmse:20.6241	validation_1-rmse:20.9624
[3]	validation_0-rmse:19.154	validation_1-rmse:19.5037
[4]	validation_0-rmse:17.9809	validation_1-rmse:18.3428
[5]	validation_0-rmse:16.9519	validation_1-rmse:17.3283
[6]	validation_0-rmse:15.9537	validation_1-rmse:16.3455
[7]	validation_0-rmse:15.1909	validation_1-rmse:15.599
[8]	validation_0-rmse:14.453	validation_1-rmse:14.8745
[9]	validation_0-rmse:13.8178	validation_1-rmse:14.2536
[10]	validation_0-rmse:13.2608	validation_1-rmse:13.7169
[11]	validation_0-rmse:12.7997	validation_1-rmse:13.2705
[12]	validation_0-rmse:12.4062	validation_1-rmse:12.8922
[13]	validation_0-rmse:12.1226	validation_1-rmse:12.6197
[14]	validation_0-rmse:11.8739	validation_1-rmse:12.379
[15]	validation_0-rmse:11.6398	validation_1-rmse:12.1547
[16]	validation_0-rmse:11.4587	validation_1-rmse:11.9846
[17]	validation_0-rmse:11.2875	validation_1-rmse:11.8238
[18]	validation_0-rmse:11.162	validation_1-rmse:11.7032
[19]	validation_0-rmse:11.0414	validation_1-rmse:11.5881
[20]	validation_0-rmse:10.9608	validation_1-rmse:11.5144
[21]	validation_0-rmse:10.8618	validation_1-rmse:11.4344
[22]	validation_0-rmse:10.7896	validation_1-rmse:11.3681
[23]	validation_0-rmse:10.7213	validation_1-rmse:11.3066
[24]	validation_0-rmse:10.6589	validation_1-rmse:11.2533
[25]	validation_0-rmse:10.607	validation_1-rmse:11.2103
[26]	validation_0-rmse:10.5646	validation_1-rmse:11.1735
[27]	validation_0-rmse:10.529	validation_1-rmse:11.1477
[28]	validation_0-rmse:10.4883	validation_1-rmse:11.123
[29]	validation_0-rmse:10.4578	validation_1-rmse:11.0998
[30]	validation_0-rmse:10.433	validation_1-rmse:11.0809
[31]	validation_0-rmse:10.4012	validation_1-rmse:11.0659
[32]	validation_0-rmse:10.3858	validation_1-rmse:11.0553
[33]	validation_0-rmse:10.3689	validation_1-rmse:11.0447
[34]	validation_0-rmse:10.3551	validation_1-rmse:11.0346
[35]	validation_0-rmse:10.3435	validation_1-rmse:11.0259
[36]	validation_0-rmse:10.3311	validation_1-rmse:11.0179
[37]	validation_0-rmse:10.3216	validation_1-rmse:11.0107
[38]	validation_0-rmse:10.3089	validation_1-rmse:11.005
[39]	validation_0-rmse:10.3019	validation_1-rmse:10.9999
[40]	validation_0-rmse:10.2794	validation_1-rmse:11.0003
[41]	validation_0-rmse:10.2683	validation_1-rmse:10.9962
[42]	validation_0-rmse:10.2631	validation_1-rmse:10.9926
[43]	validation_0-rmse:10.256	validation_1-rmse:10.9881
[44]	validation_0-rmse:10.2452	validation_1-rmse:10.9851
[45]	validation_0-rmse:10.2403	validation_1-rmse:10.9853
[46]	validation_0-rmse:10.2338	validation_1-rmse:10.9813
[47]	validation_0-rmse:10.2218	validation_1-rmse:10.972
[48]	validation_0-rmse:10.215	validation_1-rmse:10.9679
[49]	validation_0-rmse:10.1961	validation_1-rmse:10.9641
[50]	validation_0-rmse:10.189	validation_1-rmse:10.963
[51]	validation_0-rmse:10.1765	validation_1-rmse:10.9559
[52]	validation_0-rmse:10.1702	validation_1-rmse:10.953
[53]	validation_0-rmse:10.1671	validation_1-rmse:10.9497
[54]	validation_0-rmse:10.1569	validation_1-rmse:10.9482
[55]	validation_0-rmse:10.1512	validation_1-rmse:10.9465
[56]	validation_0-rmse:10.1461	validation_1-rmse:10.9502
[57]	validation_0-rmse:10.1375	validation_1-rmse:10.9482
[58]	validation_0-rmse:10.1337	validation_1-rmse:10.9462
[59]	validation_0-rmse:10.1286	validation_1-rmse:10.9439
[60]	validation_0-rmse:10.1232	validation_1-rmse:10.9418
[61]	validation_0-rmse:10.1138	validation_1-rmse:10.9428
[62]	validation_0-rmse:10.1095	validation_1-rmse:10.9415
[63]	validation_0-rmse:10.103	validation_1-rmse:10.9397
[64]	validation_0-rmse:10.0972	validation_1-rmse:10.9371
[65]	validation_0-rmse:10.0931	validation_1-rmse:10.9359
[66]	validation_0-rmse:10.0893	validation_1-rmse:10.9327
[67]	validation_0-rmse:10.0837	validation_1-rmse:10.9313
[68]	validation_0-rmse:10.073	validation_1-rmse:10.9285
[69]	validation_0-rmse:10.0637	validation_1-rmse:10.9275
[70]	validation_0-rmse:10.0557	validation_1-rmse:10.9278
[71]	validation_0-rmse:10.045	validation_1-rmse:10.9262
[72]	validation_0-rmse:10.0346	validation_1-rmse:10.9262
[73]	validation_0-rmse:10.028	validation_1-rmse:10.9269
[74]	validation_0-rmse:10.0236	validation_1-rmse:10.9282
[75]	validation_0-rmse:10.0189	validation_1-rmse:10.9313
[76]	validation_0-rmse:10.0103	validation_1-rmse:10.931
[77]	validation_0-rmse:10.0061	validation_1-rmse:10.9308
[78]	validation_0-rmse:10.0026	validation_1-rmse:10.9301
[79]	validation_0-rmse:9.9928	validation_1-rmse:10.9271
[80]	validation_0-rmse:9.98766	validation_1-rmse:10.9244
[81]	validation_0-rmse:9.97597	validation_1-rmse:10.9191
[82]	validation_0-rmse:9.97325	validation_1-rmse:10.9199
[83]	validation_0-rmse:9.96687	validation_1-rmse:10.9182
[84]	validation_0-rmse:9.96374	validation_1-rmse:10.9193
[85]	validation_0-rmse:9.96202	validation_1-rmse:10.9188
[86]	validation_0-rmse:9.95677	validation_1-rmse:10.9155
[87]	validation_0-rmse:9.95086	validation_1-rmse:10.9108
[88]	validation_0-rmse:9.94638	validation_1-rmse:10.9116
[89]	validation_0-rmse:9.94335	validation_1-rmse:10.9113
[90]	validation_0-rmse:9.93693	validation_1-rmse:10.9078
[91]	validation_0-rmse:9.93162	validation_1-rmse:10.9085
[92]	validation_0-rmse:9.92613	validation_1-rmse:10.9063
[93]	validation_0-rmse:9.92201	validation_1-rmse:10.9089
[94]	validation_0-rmse:9.91908	validation_1-rmse:10.9126
[95]	validation_0-rmse:9.91779	validation_1-rmse:10.9123
[96]	validation_0-rmse:9.9148	validation_1-rmse:10.9121
[97]	validation_0-rmse:9.90838	validation_1-rmse:10.9081
[98]	validation_0-rmse:9.89901	validation_1-rmse:10.901
[99]	validation_0-rmse:9.89527	validation_1-rmse:10.8989
[100]	validation_0-rmse:9.89203	validation_1-rmse:10.8988
[101]	validation_0-rmse:9.8891	validation_1-rmse:10.8991
[102]	validation_0-rmse:9.88823	validation_1-rmse:10.8991
[103]	validation_0-rmse:9.88612	validation_1-rmse:10.8981
[104]	validation_0-rmse:9.87749	validation_1-rmse:10.8998
[105]	validation_0-rmse:9.87472	validation_1-rmse:10.8989
[106]	validation_0-rmse:9.87226	validation_1-rmse:10.8989
[107]	validation_0-rmse:9.86391	validation_1-rmse:10.8978
[108]	validation_0-rmse:9.85925	validation_1-rmse:10.8971
[109]	validation_0-rmse:9.85783	validation_1-rmse:10.8998
[110]	validation_0-rmse:9.85064	validation_1-rmse:10.8992
[111]	validation_0-rmse:9.84544	validation_1-rmse:10.8967
[112]	validation_0-rmse:9.84166	validation_1-rmse:10.8936
[113]	validation_0-rmse:9.83435	validation_1-rmse:10.8913
[114]	validation_0-rmse:9.82567	validation_1-rmse:10.8915
[115]	validation_0-rmse:9.82201	validation_1-rmse:10.8905
[116]	validation_0-rmse:9.82182	validation_1-rmse:10.8904
[117]	validation_0-rmse:9.81434	validation_1-rmse:10.8886
[118]	validation_0-rmse:9.81086	validation_1-rmse:10.8884
[119]	validation_0-rmse:9.80892	validation_1-rmse:10.8872
[120]	validation_0-rmse:9.80716	validation_1-rmse:10.888
[121]	validation_0-rmse:9.80565	validation_1-rmse:10.8878
[122]	validation_0-rmse:9.80263	validation_1-rmse:10.8868
[123]	validation_0-rmse:9.80041	validation_1-rmse:10.8871
[124]	validation_0-rmse:9.79672	validation_1-rmse:10.8874
[125]	validation_0-rmse:9.79356	validation_1-rmse:10.8854
[126]	validation_0-rmse:9.79158	validation_1-rmse:10.8861
[127]	validation_0-rmse:9.79023	validation_1-rmse:10.8868
[128]	validation_0-rmse:9.77603	validation_1-rmse:10.8884
[129]	validation_0-rmse:9.77323	validation_1-rmse:10.8891
[130]	validation_0-rmse:9.77238	validation_1-rmse:10.8897
[131]	validation_0-rmse:9.76774	validation_1-rmse:10.8898
[132]	validation_0-rmse:9.76734	validation_1-rmse:10.8902
[133]	validation_0-rmse:9.76105	validation_1-rmse:10.8868
[134]	validation_0-rmse:9.75832	validation_1-rmse:10.8872
[135]	validation_0-rmse:9.75613	validation_1-rmse:10.8867
[136]	validation_0-rmse:9.75284	validation_1-rmse:10.8853
[137]	validation_0-rmse:9.75216	validation_1-rmse:10.8868
[138]	validation_0-rmse:9.74487	validation_1-rmse:10.8842
[139]	validation_0-rmse:9.74203	validation_1-rmse:10.8878
[140]	validation_0-rmse:9.74006	validation_1-rmse:10.887
[141]	validation_0-rmse:9.73782	validation_1-rmse:10.8863
[142]	validation_0-rmse:9.73327	validation_1-rmse:10.886
[143]	validation_0-rmse:9.73085	validation_1-rmse:10.8852
[144]	validation_0-rmse:9.72619	validation_1-rmse:10.8826
[145]	validation_0-rmse:9.71835	validation_1-rmse:10.8878
[146]	validation_0-rmse:9.71375	validation_1-rmse:10.8866
[147]	validation_0-rmse:9.70981	validation_1-rmse:10.8845
[148]	validation_0-rmse:9.70858	validation_1-rmse:10.8849
[149]	validation_0-rmse:9.70759	validation_1-rmse:10.8848
[150]	validation_0-rmse:9.70468	validation_1-rmse:10.8846
[151]	validation_0-rmse:9.70502	validation_1-rmse:10.8843
[152]	validation_0-rmse:9.70248	validation_1-rmse:10.8832
[153]	validation_0-rmse:9.70192	validation_1-rmse:10.8832
[154]	validation_0-rmse:9.69837	validation_1-rmse:10.8836
[155]	validation_0-rmse:9.69418	validation_1-rmse:10.8812
[156]	validation_0-rmse:9.68593	validation_1-rmse:10.8804
[157]	validation_0-rmse:9.6834	validation_1-rmse:10.8794
[158]	validation_0-rmse:9.67871	validation_1-rmse:10.8787
[159]	validation_0-rmse:9.67772	validation_1-rmse:10.8784
[160]	validation_0-rmse:9.6755	validation_1-rmse:10.877
[161]	validation_0-rmse:9.67147	validation_1-rmse:10.8765
[162]	validation_0-rmse:9.67164	validation_1-rmse:10.8764
[163]	validation_0-rmse:9.67005	validation_1-rmse:10.8744
[164]	validation_0-rmse:9.66623	validation_1-rmse:10.8751
[165]	validation_0-rmse:9.66213	validation_1-rmse:10.8756
[166]	validation_0-rmse:9.66181	validation_1-rmse:10.8764
[167]	validation_0-rmse:9.66074	validation_1-rmse:10.8777
[168]	validation_0-rmse:9.65571	validation_1-rmse:10.8761
[169]	validation_0-rmse:9.65542	validation_1-rmse:10.8755
[170]	validation_0-rmse:9.65484	validation_1-rmse:10.8747
[171]	validation_0-rmse:9.64747	validation_1-rmse:10.8727
[172]	validation_0-rmse:9.6467	validation_1-rmse:10.8729
[173]	validation_0-rmse:9.64116	validation_1-rmse:10.8739
[174]	validation_0-rmse:9.6386	validation_1-rmse:10.8734
[175]	validation_0-rmse:9.63874	validation_1-rmse:10.8728
[176]	validation_0-rmse:9.63429	validation_1-rmse:10.8707
[177]	validation_0-rmse:9.62935	validation_1-rmse:10.8707
[178]	validation_0-rmse:9.62062	validation_1-rmse:10.8728
[179]	validation_0-rmse:9.61938	validation_1-rmse:10.8731
[180]	validation_0-rmse:9.61658	validation_1-rmse:10.8739
[181]	validation_0-rmse:9.61327	validation_1-rmse:10.8727
[182]	validation_0-rmse:9.61164	validation_1-rmse:10.8724
[183]	validation_0-rmse:9.61053	validation_1-rmse:10.872
[184]	validation_0-rmse:9.61005	validation_1-rmse:10.8718
[185]	validation_0-rmse:9.60835	validation_1-rmse:10.8737
[186]	validation_0-rmse:9.60712	validation_1-rmse:10.8731
[187]	validation_0-rmse:9.60701	validation_1-rmse:10.8732
[188]	validation_0-rmse:9.60518	validation_1-rmse:10.8766
[189]	validation_0-rmse:9.60305	validation_1-rmse:10.8738
[190]	validation_0-rmse:9.60296	validation_1-rmse:10.8738
[191]	validation_0-rmse:9.5992	validation_1-rmse:10.8723
[192]	validation_0-rmse:9.59911	validation_1-rmse:10.8723
[193]	validation_0-rmse:9.59317	validation_1-rmse:10.8742
[194]	validation_0-rmse:9.58909	validation_1-rmse:10.8722
[195]	validation_0-rmse:9.58902	validation_1-rmse:10.8721
[196]	validation_0-rmse:9.58365	validation_1-rmse:10.8718
[197]	validation_0-rmse:9.58297	validation_1-rmse:10.872
[198]	validation_0-rmse:9.58138	validation_1-rmse:10.8721
[199]	validation_0-rmse:9.58043	validation_1-rmse:10.871
[200]	validation_0-rmse:9.57808	validation_1-rmse:10.8696
[201]	validation_0-rmse:9.57211	validation_1-rmse:10.8748
[202]	validation_0-rmse:9.57206	validation_1-rmse:10.8748
[203]	validation_0-rmse:9.57082	validation_1-rmse:10.8757
[204]	validation_0-rmse:9.56746	validation_1-rmse:10.8735
[205]	validation_0-rmse:9.56515	validation_1-rmse:10.8729
[206]	validation_0-rmse:9.5628	validation_1-rmse:10.8745
[207]	validation_0-rmse:9.55735	validation_1-rmse:10.8763
[208]	validation_0-rmse:9.55511	validation_1-rmse:10.8762
[209]	validation_0-rmse:9.55306	validation_1-rmse:10.8765
[210]	validation_0-rmse:9.5495	validation_1-rmse:10.8756
[211]	validation_0-rmse:9.54651	validation_1-rmse:10.8736
[212]	validation_0-rmse:9.54596	validation_1-rmse:10.8736
[213]	validation_0-rmse:9.54612	validation_1-rmse:10.8738
[214]	validation_0-rmse:9.54553	validation_1-rmse:10.8734
[215]	validation_0-rmse:9.54388	validation_1-rmse:10.8721
[216]	validation_0-rmse:9.54218	validation_1-rmse:10.8735
[217]	validation_0-rmse:9.53997	validation_1-rmse:10.873
[218]	validation_0-rmse:9.53959	validation_1-rmse:10.8727
[219]	validation_0-rmse:9.53686	validation_1-rmse:10.8714
[220]	validation_0-rmse:9.5368	validation_1-rmse:10.8714
[221]	validation_0-rmse:9.53488	validation_1-rmse:10.8713
[222]	validation_0-rmse:9.5341	validation_1-rmse:10.8713
[223]	validation_0-rmse:9.53265	validation_1-rmse:10.8718
[224]	validation_0-rmse:9.53201	validation_1-rmse:10.8716
[225]	validation_0-rmse:9.52959	validation_1-rmse:10.8727
[226]	validation_0-rmse:9.52955	validation_1-rmse:10.8727
[227]	validation_0-rmse:9.52743	validation_1-rmse:10.8729
[228]	validation_0-rmse:9.52443	validation_1-rmse:10.8728
[229]	validation_0-rmse:9.5228	validation_1-rmse:10.874
[230]	validation_0-rmse:9.51848	validation_1-rmse:10.8738
[231]	validation_0-rmse:9.51643	validation_1-rmse:10.8733
[232]	validation_0-rmse:9.51639	validation_1-rmse:10.8734
[233]	validation_0-rmse:9.51634	validation_1-rmse:10.8734
[234]	validation_0-rmse:9.51196	validation_1-rmse:10.8713
[235]	validation_0-rmse:9.50681	validation_1-rmse:10.8689
[236]	validation_0-rmse:9.50551	validation_1-rmse:10.8685
[237]	validation_0-rmse:9.50546	validation_1-rmse:10.8685
[238]	validation_0-rmse:9.50543	validation_1-rmse:10.8686
[239]	validation_0-rmse:9.50541	validation_1-rmse:10.8687
[240]	validation_0-rmse:9.50538	validation_1-rmse:10.8687
[241]	validation_0-rmse:9.50424	validation_1-rmse:10.8689
[242]	validation_0-rmse:9.50175	validation_1-rmse:10.8717
[243]	validation_0-rmse:9.49785	validation_1-rmse:10.8684
[244]	validation_0-rmse:9.49732	validation_1-rmse:10.8682
[245]	validation_0-rmse:9.49471	validation_1-rmse:10.8686
[246]	validation_0-rmse:9.49394	validation_1-rmse:10.8689
[247]	validation_0-rmse:9.49392	validation_1-rmse:10.8689
[248]	validation_0-rmse:9.49293	validation_1-rmse:10.8704
[249]	validation_0-rmse:9.49264	validation_1-rmse:10.8707
[250]	validation_0-rmse:9.49261	validation_1-rmse:10.8708
[251]	validation_0-rmse:9.48801	validation_1-rmse:10.8692
[252]	validation_0-rmse:9.48616	validation_1-rmse:10.8688
[253]	validation_0-rmse:9.48528	validation_1-rmse:10.869
[254]	validation_0-rmse:9.48472	validation_1-rmse:10.8706
[255]	validation_0-rmse:9.48471	validation_1-rmse:10.8706
[256]	validation_0-rmse:9.48469	validation_1-rmse:10.8706
[257]	validation_0-rmse:9.4782	validation_1-rmse:10.8716
[258]	validation_0-rmse:9.4769	validation_1-rmse:10.8724
[259]	validation_0-rmse:9.47205	validation_1-rmse:10.8763
[260]	validation_0-rmse:9.47062	validation_1-rmse:10.8761
[261]	validation_0-rmse:9.46656	validation_1-rmse:10.8735
[262]	validation_0-rmse:9.46251	validation_1-rmse:10.8735
[263]	validation_0-rmse:9.45969	validation_1-rmse:10.8722
[264]	validation_0-rmse:9.45815	validation_1-rmse:10.8729
[265]	validation_0-rmse:9.45813	validation_1-rmse:10.8728
[266]	validation_0-rmse:9.45602	validation_1-rmse:10.8725
[267]	validation_0-rmse:9.45431	validation_1-rmse:10.8724
[268]	validation_0-rmse:9.45196	validation_1-rmse:10.8724
[269]	validation_0-rmse:9.44936	validation_1-rmse:10.8717
[270]	validation_0-rmse:9.44934	validation_1-rmse:10.8717
[271]	validation_0-rmse:9.44863	validation_1-rmse:10.8717
[272]	validation_0-rmse:9.44619	validation_1-rmse:10.8737
[273]	validation_0-rmse:9.44617	validation_1-rmse:10.8737
[274]	validation_0-rmse:9.44616	validation_1-rmse:10.8738
[275]	validation_0-rmse:9.44509	validation_1-rmse:10.8741
[276]	validation_0-rmse:9.44449	validation_1-rmse:10.8745
[277]	validation_0-rmse:9.44312	validation_1-rmse:10.875
[278]	validation_0-rmse:9.4431	validation_1-rmse:10.875
[279]	validation_0-rmse:9.43801	validation_1-rmse:10.8755
[280]	validation_0-rmse:9.43535	validation_1-rmse:10.8758
[281]	validation_0-rmse:9.43533	validation_1-rmse:10.8759
[282]	validation_0-rmse:9.43325	validation_1-rmse:10.8745
[283]	validation_0-rmse:9.43324	validation_1-rmse:10.8746
[284]	validation_0-rmse:9.43021	validation_1-rmse:10.8754
[285]	validation_0-rmse:9.42943	validation_1-rmse:10.876
[286]	validation_0-rmse:9.42754	validation_1-rmse:10.8761
[287]	validation_0-rmse:9.42753	validation_1-rmse:10.8761
[288]	validation_0-rmse:9.42752	validation_1-rmse:10.8761
[289]	validation_0-rmse:9.42433	validation_1-rmse:10.8751
[290]	validation_0-rmse:9.42431	validation_1-rmse:10.8751
[291]	validation_0-rmse:9.41855	validation_1-rmse:10.8721
[292]	validation_0-rmse:9.41537	validation_1-rmse:10.8706
[293]	validation_0-rmse:9.41536	validation_1-rmse:10.8706
[294]	validation_0-rmse:9.41395	validation_1-rmse:10.8691
Stopping. Best iteration:
[244]	validation_0-rmse:9.49732	validation_1-rmse:10.8682

[0]	validation_0-rmse:24.2313	validation_1-rmse:24.5457
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3158	validation_1-rmse:22.6377
[2]	validation_0-rmse:20.6282	validation_1-rmse:20.9603
[3]	validation_0-rmse:19.1614	validation_1-rmse:19.5051
[4]	validation_0-rmse:17.9922	validation_1-rmse:18.3475
[5]	validation_0-rmse:16.9676	validation_1-rmse:17.3382
[6]	validation_0-rmse:15.9723	validation_1-rmse:16.3561
[7]	validation_0-rmse:15.2156	validation_1-rmse:15.6122
[8]	validation_0-rmse:14.4784	validation_1-rmse:14.8866
[9]	validation_0-rmse:13.8468	validation_1-rmse:14.267
[10]	validation_0-rmse:13.3006	validation_1-rmse:13.7305
[11]	validation_0-rmse:12.8419	validation_1-rmse:13.2848
[12]	validation_0-rmse:12.4533	validation_1-rmse:12.9101
[13]	validation_0-rmse:12.1732	validation_1-rmse:12.6425
[14]	validation_0-rmse:11.9304	validation_1-rmse:12.407
[15]	validation_0-rmse:11.6987	validation_1-rmse:12.1836
[16]	validation_0-rmse:11.5226	validation_1-rmse:12.0158
[17]	validation_0-rmse:11.3555	validation_1-rmse:11.8543
[18]	validation_0-rmse:11.2346	validation_1-rmse:11.7364
[19]	validation_0-rmse:11.1172	validation_1-rmse:11.6242
[20]	validation_0-rmse:11.0397	validation_1-rmse:11.5521
[21]	validation_0-rmse:10.951	validation_1-rmse:11.4731
[22]	validation_0-rmse:10.8818	validation_1-rmse:11.4061
[23]	validation_0-rmse:10.8197	validation_1-rmse:11.3502
[24]	validation_0-rmse:10.7643	validation_1-rmse:11.2993
[25]	validation_0-rmse:10.7104	validation_1-rmse:11.2601
[26]	validation_0-rmse:10.6704	validation_1-rmse:11.222
[27]	validation_0-rmse:10.637	validation_1-rmse:11.1924
[28]	validation_0-rmse:10.6021	validation_1-rmse:11.1706
[29]	validation_0-rmse:10.5755	validation_1-rmse:11.1488
[30]	validation_0-rmse:10.5571	validation_1-rmse:11.1316
[31]	validation_0-rmse:10.534	validation_1-rmse:11.1176
[32]	validation_0-rmse:10.5204	validation_1-rmse:11.1089
[33]	validation_0-rmse:10.5008	validation_1-rmse:11.0983
[34]	validation_0-rmse:10.4779	validation_1-rmse:11.0881
[35]	validation_0-rmse:10.467	validation_1-rmse:11.0792
[36]	validation_0-rmse:10.4555	validation_1-rmse:11.071
[37]	validation_0-rmse:10.4464	validation_1-rmse:11.0635
[38]	validation_0-rmse:10.4347	validation_1-rmse:11.0563
[39]	validation_0-rmse:10.4259	validation_1-rmse:11.0494
[40]	validation_0-rmse:10.4118	validation_1-rmse:11.0443
[41]	validation_0-rmse:10.4001	validation_1-rmse:11.0368
[42]	validation_0-rmse:10.396	validation_1-rmse:11.0344
[43]	validation_0-rmse:10.3898	validation_1-rmse:11.0316
[44]	validation_0-rmse:10.3801	validation_1-rmse:11.0271
[45]	validation_0-rmse:10.3753	validation_1-rmse:11.0268
[46]	validation_0-rmse:10.3639	validation_1-rmse:11.0207
[47]	validation_0-rmse:10.3555	validation_1-rmse:11.0153
[48]	validation_0-rmse:10.3501	validation_1-rmse:11.0162
[49]	validation_0-rmse:10.3415	validation_1-rmse:11.0121
[50]	validation_0-rmse:10.3308	validation_1-rmse:11.0167
[51]	validation_0-rmse:10.3213	validation_1-rmse:11.0119
[52]	validation_0-rmse:10.3173	validation_1-rmse:11.0101
[53]	validation_0-rmse:10.3132	validation_1-rmse:11.0076
[54]	validation_0-rmse:10.3031	validation_1-rmse:11.008
[55]	validation_0-rmse:10.2962	validation_1-rmse:11.0049
[56]	validation_0-rmse:10.2921	validation_1-rmse:11.0153
[57]	validation_0-rmse:10.2802	validation_1-rmse:11.0145
[58]	validation_0-rmse:10.2757	validation_1-rmse:11.0124
[59]	validation_0-rmse:10.2699	validation_1-rmse:11.0086
[60]	validation_0-rmse:10.2672	validation_1-rmse:11.0098
[61]	validation_0-rmse:10.2598	validation_1-rmse:11.0092
[62]	validation_0-rmse:10.2583	validation_1-rmse:11.0084
[63]	validation_0-rmse:10.2539	validation_1-rmse:11.0054
[64]	validation_0-rmse:10.2513	validation_1-rmse:11.0039
[65]	validation_0-rmse:10.249	validation_1-rmse:11.0021
[66]	validation_0-rmse:10.2446	validation_1-rmse:10.9998
[67]	validation_0-rmse:10.2392	validation_1-rmse:10.9964
[68]	validation_0-rmse:10.2322	validation_1-rmse:10.9903
[69]	validation_0-rmse:10.2249	validation_1-rmse:10.9954
[70]	validation_0-rmse:10.2137	validation_1-rmse:10.9966
[71]	validation_0-rmse:10.2068	validation_1-rmse:10.9945
[72]	validation_0-rmse:10.2008	validation_1-rmse:10.9952
[73]	validation_0-rmse:10.1914	validation_1-rmse:11.0001
[74]	validation_0-rmse:10.1894	validation_1-rmse:11.0025
[75]	validation_0-rmse:10.1844	validation_1-rmse:11.0012
[76]	validation_0-rmse:10.1719	validation_1-rmse:11.0029
[77]	validation_0-rmse:10.1635	validation_1-rmse:11.0055
[78]	validation_0-rmse:10.1604	validation_1-rmse:11.0046
[79]	validation_0-rmse:10.154	validation_1-rmse:11.0008
[80]	validation_0-rmse:10.1512	validation_1-rmse:10.9986
[81]	validation_0-rmse:10.1476	validation_1-rmse:10.997
[82]	validation_0-rmse:10.1453	validation_1-rmse:10.9981
[83]	validation_0-rmse:10.1408	validation_1-rmse:10.9975
[84]	validation_0-rmse:10.1387	validation_1-rmse:10.9979
[85]	validation_0-rmse:10.1347	validation_1-rmse:10.9951
[86]	validation_0-rmse:10.1308	validation_1-rmse:10.9927
[87]	validation_0-rmse:10.1297	validation_1-rmse:10.9926
[88]	validation_0-rmse:10.1245	validation_1-rmse:10.9901
[89]	validation_0-rmse:10.1212	validation_1-rmse:10.9901
[90]	validation_0-rmse:10.117	validation_1-rmse:10.9875
[91]	validation_0-rmse:10.111	validation_1-rmse:10.9884
[92]	validation_0-rmse:10.1087	validation_1-rmse:10.9876
[93]	validation_0-rmse:10.1083	validation_1-rmse:10.9907
[94]	validation_0-rmse:10.1053	validation_1-rmse:10.9924
[95]	validation_0-rmse:10.1038	validation_1-rmse:10.9915
[96]	validation_0-rmse:10.1029	validation_1-rmse:10.9898
[97]	validation_0-rmse:10.0991	validation_1-rmse:10.9876
[98]	validation_0-rmse:10.0935	validation_1-rmse:10.985
[99]	validation_0-rmse:10.0914	validation_1-rmse:10.9828
[100]	validation_0-rmse:10.0882	validation_1-rmse:10.9835
[101]	validation_0-rmse:10.0787	validation_1-rmse:10.9824
[102]	validation_0-rmse:10.0757	validation_1-rmse:10.9813
[103]	validation_0-rmse:10.0738	validation_1-rmse:10.9792
[104]	validation_0-rmse:10.0651	validation_1-rmse:10.9806
[105]	validation_0-rmse:10.0638	validation_1-rmse:10.9802
[106]	validation_0-rmse:10.0604	validation_1-rmse:10.9799
[107]	validation_0-rmse:10.0509	validation_1-rmse:10.9806
[108]	validation_0-rmse:10.0494	validation_1-rmse:10.9799
[109]	validation_0-rmse:10.0481	validation_1-rmse:10.9826
[110]	validation_0-rmse:10.0458	validation_1-rmse:10.9808
[111]	validation_0-rmse:10.0458	validation_1-rmse:10.9832
[112]	validation_0-rmse:10.0424	validation_1-rmse:10.9841
[113]	validation_0-rmse:10.0404	validation_1-rmse:10.983
[114]	validation_0-rmse:10.0328	validation_1-rmse:10.9855
[115]	validation_0-rmse:10.0321	validation_1-rmse:10.9865
[116]	validation_0-rmse:10.0268	validation_1-rmse:10.9826
[117]	validation_0-rmse:10.0213	validation_1-rmse:10.9807
[118]	validation_0-rmse:10.021	validation_1-rmse:10.9808
[119]	validation_0-rmse:10.02	validation_1-rmse:10.9798
[120]	validation_0-rmse:10.0184	validation_1-rmse:10.978
[121]	validation_0-rmse:10.0161	validation_1-rmse:10.9772
[122]	validation_0-rmse:10.013	validation_1-rmse:10.978
[123]	validation_0-rmse:10.0064	validation_1-rmse:10.9745
[124]	validation_0-rmse:10.0001	validation_1-rmse:10.9729
[125]	validation_0-rmse:9.99815	validation_1-rmse:10.9717
[126]	validation_0-rmse:9.99752	validation_1-rmse:10.9715
[127]	validation_0-rmse:9.99591	validation_1-rmse:10.9709
[128]	validation_0-rmse:9.99252	validation_1-rmse:10.9677
[129]	validation_0-rmse:9.99095	validation_1-rmse:10.9647
[130]	validation_0-rmse:9.99043	validation_1-rmse:10.9651
[131]	validation_0-rmse:9.98315	validation_1-rmse:10.9687
[132]	validation_0-rmse:9.98291	validation_1-rmse:10.969
[133]	validation_0-rmse:9.9767	validation_1-rmse:10.9764
[134]	validation_0-rmse:9.97439	validation_1-rmse:10.979
[135]	validation_0-rmse:9.97239	validation_1-rmse:10.9775
[136]	validation_0-rmse:9.97131	validation_1-rmse:10.9777
[137]	validation_0-rmse:9.9685	validation_1-rmse:10.9834
[138]	validation_0-rmse:9.96093	validation_1-rmse:10.9865
[139]	validation_0-rmse:9.95911	validation_1-rmse:10.9837
[140]	validation_0-rmse:9.95823	validation_1-rmse:10.9826
[141]	validation_0-rmse:9.95639	validation_1-rmse:10.9827
[142]	validation_0-rmse:9.95392	validation_1-rmse:10.9819
[143]	validation_0-rmse:9.95262	validation_1-rmse:10.9815
[144]	validation_0-rmse:9.94374	validation_1-rmse:10.9818
[145]	validation_0-rmse:9.93472	validation_1-rmse:10.9932
[146]	validation_0-rmse:9.9266	validation_1-rmse:11.0011
[147]	validation_0-rmse:9.92493	validation_1-rmse:11.0008
[148]	validation_0-rmse:9.92346	validation_1-rmse:11.0001
[149]	validation_0-rmse:9.91987	validation_1-rmse:10.9973
[150]	validation_0-rmse:9.91951	validation_1-rmse:10.9971
[151]	validation_0-rmse:9.91649	validation_1-rmse:10.9971
[152]	validation_0-rmse:9.91311	validation_1-rmse:10.9957
[153]	validation_0-rmse:9.91249	validation_1-rmse:10.9952
[154]	validation_0-rmse:9.90749	validation_1-rmse:11.0008
[155]	validation_0-rmse:9.90307	validation_1-rmse:11.0068
[156]	validation_0-rmse:9.8967	validation_1-rmse:11.0095
[157]	validation_0-rmse:9.89545	validation_1-rmse:11.0092
[158]	validation_0-rmse:9.89151	validation_1-rmse:11.0074
[159]	validation_0-rmse:9.8913	validation_1-rmse:11.0082
[160]	validation_0-rmse:9.8908	validation_1-rmse:11.008
[161]	validation_0-rmse:9.88748	validation_1-rmse:11.0064
[162]	validation_0-rmse:9.88473	validation_1-rmse:11.0073
[163]	validation_0-rmse:9.87902	validation_1-rmse:11.0041
[164]	validation_0-rmse:9.87688	validation_1-rmse:11.004
[165]	validation_0-rmse:9.86896	validation_1-rmse:11.0077
[166]	validation_0-rmse:9.86743	validation_1-rmse:11.0102
[167]	validation_0-rmse:9.86692	validation_1-rmse:11.0108
[168]	validation_0-rmse:9.86332	validation_1-rmse:11.0092
[169]	validation_0-rmse:9.8624	validation_1-rmse:11.0077
[170]	validation_0-rmse:9.86138	validation_1-rmse:11.0072
[171]	validation_0-rmse:9.85275	validation_1-rmse:11.0083
[172]	validation_0-rmse:9.85194	validation_1-rmse:11.0089
[173]	validation_0-rmse:9.84566	validation_1-rmse:11.0055
[174]	validation_0-rmse:9.84335	validation_1-rmse:11.005
[175]	validation_0-rmse:9.837	validation_1-rmse:11.0026
[176]	validation_0-rmse:9.83468	validation_1-rmse:11.0016
[177]	validation_0-rmse:9.83048	validation_1-rmse:11.0005
[178]	validation_0-rmse:9.82518	validation_1-rmse:11.0043
[179]	validation_0-rmse:9.82464	validation_1-rmse:11.0035
Stopping. Best iteration:
[129]	validation_0-rmse:9.99095	validation_1-rmse:10.9647

[0]	validation_0-rmse:24.2239	validation_1-rmse:24.5471
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3032	validation_1-rmse:22.6371
[2]	validation_0-rmse:20.6091	validation_1-rmse:20.9567
[3]	validation_0-rmse:19.1189	validation_1-rmse:19.4876
[4]	validation_0-rmse:17.8164	validation_1-rmse:18.2111
[5]	validation_0-rmse:16.6811	validation_1-rmse:17.0985
[6]	validation_0-rmse:15.7	validation_1-rmse:16.1407
[7]	validation_0-rmse:14.8539	validation_1-rmse:15.3163
[8]	validation_0-rmse:14.1367	validation_1-rmse:14.6217
[9]	validation_0-rmse:13.5192	validation_1-rmse:14.0288
[10]	validation_0-rmse:12.9801	validation_1-rmse:13.5137
[11]	validation_0-rmse:12.5293	validation_1-rmse:13.0846
[12]	validation_0-rmse:12.141	validation_1-rmse:12.7237
[13]	validation_0-rmse:11.8111	validation_1-rmse:12.4233
[14]	validation_0-rmse:11.5312	validation_1-rmse:12.176
[15]	validation_0-rmse:11.3076	validation_1-rmse:11.9698
[16]	validation_0-rmse:11.106	validation_1-rmse:11.7977
[17]	validation_0-rmse:10.9454	validation_1-rmse:11.654
[18]	validation_0-rmse:10.8073	validation_1-rmse:11.5306
[19]	validation_0-rmse:10.6965	validation_1-rmse:11.4326
[20]	validation_0-rmse:10.5998	validation_1-rmse:11.3482
[21]	validation_0-rmse:10.5126	validation_1-rmse:11.2791
[22]	validation_0-rmse:10.4319	validation_1-rmse:11.2261
[23]	validation_0-rmse:10.3703	validation_1-rmse:11.179
[24]	validation_0-rmse:10.305	validation_1-rmse:11.1359
[25]	validation_0-rmse:10.2574	validation_1-rmse:11.0992
[26]	validation_0-rmse:10.2156	validation_1-rmse:11.0692
[27]	validation_0-rmse:10.177	validation_1-rmse:11.0509
[28]	validation_0-rmse:10.1315	validation_1-rmse:11.0335
[29]	validation_0-rmse:10.1029	validation_1-rmse:11.0139
[30]	validation_0-rmse:10.0711	validation_1-rmse:11.0035
[31]	validation_0-rmse:10.0495	validation_1-rmse:11.0003
[32]	validation_0-rmse:10.0268	validation_1-rmse:10.9853
[33]	validation_0-rmse:10.0095	validation_1-rmse:10.9753
[34]	validation_0-rmse:9.98698	validation_1-rmse:10.9638
[35]	validation_0-rmse:9.97096	validation_1-rmse:10.9577
[36]	validation_0-rmse:9.95254	validation_1-rmse:10.9522
[37]	validation_0-rmse:9.94205	validation_1-rmse:10.9457
[38]	validation_0-rmse:9.92538	validation_1-rmse:10.9428
[39]	validation_0-rmse:9.91331	validation_1-rmse:10.9378
[40]	validation_0-rmse:9.90249	validation_1-rmse:10.9349
[41]	validation_0-rmse:9.89179	validation_1-rmse:10.9346
[42]	validation_0-rmse:9.8812	validation_1-rmse:10.9278
[43]	validation_0-rmse:9.8717	validation_1-rmse:10.9213
[44]	validation_0-rmse:9.86656	validation_1-rmse:10.9219
[45]	validation_0-rmse:9.85728	validation_1-rmse:10.9197
[46]	validation_0-rmse:9.84657	validation_1-rmse:10.9192
[47]	validation_0-rmse:9.83559	validation_1-rmse:10.9131
[48]	validation_0-rmse:9.82592	validation_1-rmse:10.9102
[49]	validation_0-rmse:9.81566	validation_1-rmse:10.9078
[50]	validation_0-rmse:9.80984	validation_1-rmse:10.914
[51]	validation_0-rmse:9.79429	validation_1-rmse:10.9191
[52]	validation_0-rmse:9.78306	validation_1-rmse:10.9147
[53]	validation_0-rmse:9.78025	validation_1-rmse:10.9137
[54]	validation_0-rmse:9.76472	validation_1-rmse:10.9154
[55]	validation_0-rmse:9.75103	validation_1-rmse:10.9225
[56]	validation_0-rmse:9.74456	validation_1-rmse:10.9229
[57]	validation_0-rmse:9.72933	validation_1-rmse:10.9218
[58]	validation_0-rmse:9.7253	validation_1-rmse:10.9194
[59]	validation_0-rmse:9.71591	validation_1-rmse:10.9115
[60]	validation_0-rmse:9.70947	validation_1-rmse:10.9094
[61]	validation_0-rmse:9.70156	validation_1-rmse:10.9132
[62]	validation_0-rmse:9.69747	validation_1-rmse:10.9114
[63]	validation_0-rmse:9.69317	validation_1-rmse:10.9126
[64]	validation_0-rmse:9.69202	validation_1-rmse:10.9121
[65]	validation_0-rmse:9.6866	validation_1-rmse:10.911
[66]	validation_0-rmse:9.68104	validation_1-rmse:10.9062
[67]	validation_0-rmse:9.67311	validation_1-rmse:10.9018
[68]	validation_0-rmse:9.65963	validation_1-rmse:10.8994
[69]	validation_0-rmse:9.65088	validation_1-rmse:10.9031
[70]	validation_0-rmse:9.64188	validation_1-rmse:10.9076
[71]	validation_0-rmse:9.63834	validation_1-rmse:10.9069
[72]	validation_0-rmse:9.63312	validation_1-rmse:10.9147
[73]	validation_0-rmse:9.62781	validation_1-rmse:10.9155
[74]	validation_0-rmse:9.6228	validation_1-rmse:10.9174
[75]	validation_0-rmse:9.61403	validation_1-rmse:10.917
[76]	validation_0-rmse:9.60033	validation_1-rmse:10.9148
[77]	validation_0-rmse:9.58994	validation_1-rmse:10.9186
[78]	validation_0-rmse:9.57973	validation_1-rmse:10.9161
[79]	validation_0-rmse:9.57952	validation_1-rmse:10.9162
[80]	validation_0-rmse:9.57558	validation_1-rmse:10.9141
[81]	validation_0-rmse:9.56618	validation_1-rmse:10.9167
[82]	validation_0-rmse:9.5585	validation_1-rmse:10.9168
[83]	validation_0-rmse:9.54907	validation_1-rmse:10.9135
[84]	validation_0-rmse:9.54661	validation_1-rmse:10.9102
[85]	validation_0-rmse:9.5387	validation_1-rmse:10.918
[86]	validation_0-rmse:9.53357	validation_1-rmse:10.9147
[87]	validation_0-rmse:9.51962	validation_1-rmse:10.908
[88]	validation_0-rmse:9.51594	validation_1-rmse:10.9051
[89]	validation_0-rmse:9.51325	validation_1-rmse:10.9028
[90]	validation_0-rmse:9.50511	validation_1-rmse:10.8987
[91]	validation_0-rmse:9.49343	validation_1-rmse:10.9015
[92]	validation_0-rmse:9.4872	validation_1-rmse:10.8964
[93]	validation_0-rmse:9.48484	validation_1-rmse:10.8975
[94]	validation_0-rmse:9.48236	validation_1-rmse:10.8963
[95]	validation_0-rmse:9.47655	validation_1-rmse:10.8936
[96]	validation_0-rmse:9.47468	validation_1-rmse:10.8927
[97]	validation_0-rmse:9.47129	validation_1-rmse:10.8916
[98]	validation_0-rmse:9.45886	validation_1-rmse:10.8875
[99]	validation_0-rmse:9.45418	validation_1-rmse:10.885
[100]	validation_0-rmse:9.44948	validation_1-rmse:10.8846
[101]	validation_0-rmse:9.43871	validation_1-rmse:10.8918
[102]	validation_0-rmse:9.43249	validation_1-rmse:10.8973
[103]	validation_0-rmse:9.42479	validation_1-rmse:10.8956
[104]	validation_0-rmse:9.41878	validation_1-rmse:10.8972
[105]	validation_0-rmse:9.41348	validation_1-rmse:10.8958
[106]	validation_0-rmse:9.41097	validation_1-rmse:10.9009
[107]	validation_0-rmse:9.40796	validation_1-rmse:10.9006
[108]	validation_0-rmse:9.40321	validation_1-rmse:10.9011
[109]	validation_0-rmse:9.40078	validation_1-rmse:10.8994
[110]	validation_0-rmse:9.39871	validation_1-rmse:10.899
[111]	validation_0-rmse:9.39555	validation_1-rmse:10.9004
[112]	validation_0-rmse:9.38935	validation_1-rmse:10.8992
[113]	validation_0-rmse:9.38041	validation_1-rmse:10.8959
[114]	validation_0-rmse:9.37136	validation_1-rmse:10.9003
[115]	validation_0-rmse:9.3696	validation_1-rmse:10.9006
[116]	validation_0-rmse:9.35668	validation_1-rmse:10.8971
[117]	validation_0-rmse:9.34985	validation_1-rmse:10.8928
[118]	validation_0-rmse:9.34698	validation_1-rmse:10.8908
[119]	validation_0-rmse:9.34526	validation_1-rmse:10.8899
[120]	validation_0-rmse:9.33837	validation_1-rmse:10.8881
[121]	validation_0-rmse:9.33471	validation_1-rmse:10.8893
[122]	validation_0-rmse:9.32962	validation_1-rmse:10.8921
[123]	validation_0-rmse:9.3218	validation_1-rmse:10.8911
[124]	validation_0-rmse:9.31442	validation_1-rmse:10.8899
[125]	validation_0-rmse:9.31044	validation_1-rmse:10.8911
[126]	validation_0-rmse:9.30874	validation_1-rmse:10.8919
[127]	validation_0-rmse:9.30496	validation_1-rmse:10.8906
[128]	validation_0-rmse:9.29815	validation_1-rmse:10.8869
[129]	validation_0-rmse:9.28978	validation_1-rmse:10.8854
[130]	validation_0-rmse:9.28632	validation_1-rmse:10.8864
[131]	validation_0-rmse:9.27922	validation_1-rmse:10.8871
[132]	validation_0-rmse:9.27813	validation_1-rmse:10.8888
[133]	validation_0-rmse:9.2698	validation_1-rmse:10.8921
[134]	validation_0-rmse:9.26283	validation_1-rmse:10.8909
[135]	validation_0-rmse:9.26272	validation_1-rmse:10.8909
[136]	validation_0-rmse:9.25773	validation_1-rmse:10.8893
[137]	validation_0-rmse:9.25573	validation_1-rmse:10.8958
[138]	validation_0-rmse:9.25256	validation_1-rmse:10.8951
[139]	validation_0-rmse:9.25099	validation_1-rmse:10.8939
[140]	validation_0-rmse:9.24683	validation_1-rmse:10.8905
[141]	validation_0-rmse:9.24221	validation_1-rmse:10.8905
[142]	validation_0-rmse:9.2368	validation_1-rmse:10.8905
[143]	validation_0-rmse:9.23019	validation_1-rmse:10.8895
[144]	validation_0-rmse:9.22666	validation_1-rmse:10.888
[145]	validation_0-rmse:9.22397	validation_1-rmse:10.8941
[146]	validation_0-rmse:9.2197	validation_1-rmse:10.8923
[147]	validation_0-rmse:9.21649	validation_1-rmse:10.8934
[148]	validation_0-rmse:9.20597	validation_1-rmse:10.8877
[149]	validation_0-rmse:9.20359	validation_1-rmse:10.8834
[150]	validation_0-rmse:9.2011	validation_1-rmse:10.8833
[151]	validation_0-rmse:9.19094	validation_1-rmse:10.8781
[152]	validation_0-rmse:9.18504	validation_1-rmse:10.8782
[153]	validation_0-rmse:9.18173	validation_1-rmse:10.8781
[154]	validation_0-rmse:9.17187	validation_1-rmse:10.8802
[155]	validation_0-rmse:9.16405	validation_1-rmse:10.8807
[156]	validation_0-rmse:9.16299	validation_1-rmse:10.8795
[157]	validation_0-rmse:9.15671	validation_1-rmse:10.8823
[158]	validation_0-rmse:9.15477	validation_1-rmse:10.8843
[159]	validation_0-rmse:9.15005	validation_1-rmse:10.8851
[160]	validation_0-rmse:9.14211	validation_1-rmse:10.8905
[161]	validation_0-rmse:9.13856	validation_1-rmse:10.8907
[162]	validation_0-rmse:9.13774	validation_1-rmse:10.8907
[163]	validation_0-rmse:9.13438	validation_1-rmse:10.8896
[164]	validation_0-rmse:9.13096	validation_1-rmse:10.8903
[165]	validation_0-rmse:9.12673	validation_1-rmse:10.8944
[166]	validation_0-rmse:9.12348	validation_1-rmse:10.8939
[167]	validation_0-rmse:9.11902	validation_1-rmse:10.8912
[168]	validation_0-rmse:9.11135	validation_1-rmse:10.8893
[169]	validation_0-rmse:9.1099	validation_1-rmse:10.8883
[170]	validation_0-rmse:9.10928	validation_1-rmse:10.888
[171]	validation_0-rmse:9.10277	validation_1-rmse:10.889
[172]	validation_0-rmse:9.09934	validation_1-rmse:10.8892
[173]	validation_0-rmse:9.09258	validation_1-rmse:10.8859
[174]	validation_0-rmse:9.08784	validation_1-rmse:10.8882
[175]	validation_0-rmse:9.08338	validation_1-rmse:10.8899
[176]	validation_0-rmse:9.08074	validation_1-rmse:10.8908
[177]	validation_0-rmse:9.07539	validation_1-rmse:10.8971
[178]	validation_0-rmse:9.07353	validation_1-rmse:10.8966
[179]	validation_0-rmse:9.06762	validation_1-rmse:10.8961
[180]	validation_0-rmse:9.06488	validation_1-rmse:10.8961
[181]	validation_0-rmse:9.06267	validation_1-rmse:10.8948
[182]	validation_0-rmse:9.06034	validation_1-rmse:10.8964
[183]	validation_0-rmse:9.05804	validation_1-rmse:10.8961
[184]	validation_0-rmse:9.05579	validation_1-rmse:10.8955
[185]	validation_0-rmse:9.0528	validation_1-rmse:10.8977
[186]	validation_0-rmse:9.04366	validation_1-rmse:10.8955
[187]	validation_0-rmse:9.03737	validation_1-rmse:10.8929
[188]	validation_0-rmse:9.03535	validation_1-rmse:10.8906
[189]	validation_0-rmse:9.03505	validation_1-rmse:10.8887
[190]	validation_0-rmse:9.03499	validation_1-rmse:10.8887
[191]	validation_0-rmse:9.03079	validation_1-rmse:10.8885
[192]	validation_0-rmse:9.02561	validation_1-rmse:10.8872
[193]	validation_0-rmse:9.01941	validation_1-rmse:10.8871
[194]	validation_0-rmse:9.01516	validation_1-rmse:10.8863
[195]	validation_0-rmse:9.00784	validation_1-rmse:10.8899
[196]	validation_0-rmse:9.00564	validation_1-rmse:10.89
[197]	validation_0-rmse:9.00457	validation_1-rmse:10.8897
[198]	validation_0-rmse:9.00299	validation_1-rmse:10.8901
[199]	validation_0-rmse:9.00035	validation_1-rmse:10.89
[200]	validation_0-rmse:8.99993	validation_1-rmse:10.8905
[201]	validation_0-rmse:8.99235	validation_1-rmse:10.8909
Stopping. Best iteration:
[151]	validation_0-rmse:9.19094	validation_1-rmse:10.8781

[0]	validation_0-rmse:24.2377	validation_1-rmse:24.5519
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3274	validation_1-rmse:22.6482
[2]	validation_0-rmse:20.649	validation_1-rmse:20.9779
[3]	validation_0-rmse:19.1818	validation_1-rmse:19.5188
[4]	validation_0-rmse:17.9075	validation_1-rmse:18.2532
[5]	validation_0-rmse:16.7965	validation_1-rmse:17.1492
[6]	validation_0-rmse:15.8367	validation_1-rmse:16.198
[7]	validation_0-rmse:15.0173	validation_1-rmse:15.3873
[8]	validation_0-rmse:14.3175	validation_1-rmse:14.6963
[9]	validation_0-rmse:13.7183	validation_1-rmse:14.1058
[10]	validation_0-rmse:13.2104	validation_1-rmse:13.6071
[11]	validation_0-rmse:12.7833	validation_1-rmse:13.1873
[12]	validation_0-rmse:12.4247	validation_1-rmse:12.8366
[13]	validation_0-rmse:12.1243	validation_1-rmse:12.5434
[14]	validation_0-rmse:11.8751	validation_1-rmse:12.3007
[15]	validation_0-rmse:11.6691	validation_1-rmse:12.0992
[16]	validation_0-rmse:11.4949	validation_1-rmse:11.9322
[17]	validation_0-rmse:11.3497	validation_1-rmse:11.7906
[18]	validation_0-rmse:11.2296	validation_1-rmse:11.6755
[19]	validation_0-rmse:11.1288	validation_1-rmse:11.5779
[20]	validation_0-rmse:11.046	validation_1-rmse:11.4972
[21]	validation_0-rmse:10.9744	validation_1-rmse:11.4304
[22]	validation_0-rmse:10.9147	validation_1-rmse:11.3718
[23]	validation_0-rmse:10.8636	validation_1-rmse:11.3241
[24]	validation_0-rmse:10.823	validation_1-rmse:11.2851
[25]	validation_0-rmse:10.7849	validation_1-rmse:11.2485
[26]	validation_0-rmse:10.7564	validation_1-rmse:11.2225
[27]	validation_0-rmse:10.7298	validation_1-rmse:11.1965
[28]	validation_0-rmse:10.707	validation_1-rmse:11.1774
[29]	validation_0-rmse:10.6838	validation_1-rmse:11.1578
[30]	validation_0-rmse:10.6644	validation_1-rmse:11.1404
[31]	validation_0-rmse:10.6487	validation_1-rmse:11.127
[32]	validation_0-rmse:10.6335	validation_1-rmse:11.1147
[33]	validation_0-rmse:10.618	validation_1-rmse:11.1041
[34]	validation_0-rmse:10.6079	validation_1-rmse:11.0962
[35]	validation_0-rmse:10.598	validation_1-rmse:11.0905
[36]	validation_0-rmse:10.5895	validation_1-rmse:11.0837
[37]	validation_0-rmse:10.5811	validation_1-rmse:11.0772
[38]	validation_0-rmse:10.5738	validation_1-rmse:11.0727
[39]	validation_0-rmse:10.5672	validation_1-rmse:11.0667
[40]	validation_0-rmse:10.5543	validation_1-rmse:11.0598
[41]	validation_0-rmse:10.5464	validation_1-rmse:11.0584
[42]	validation_0-rmse:10.5386	validation_1-rmse:11.0533
[43]	validation_0-rmse:10.5329	validation_1-rmse:11.048
[44]	validation_0-rmse:10.5263	validation_1-rmse:11.0436
[45]	validation_0-rmse:10.5195	validation_1-rmse:11.0414
[46]	validation_0-rmse:10.5146	validation_1-rmse:11.0401
[47]	validation_0-rmse:10.5103	validation_1-rmse:11.0377
[48]	validation_0-rmse:10.4971	validation_1-rmse:11.0325
[49]	validation_0-rmse:10.4933	validation_1-rmse:11.0296
[50]	validation_0-rmse:10.4891	validation_1-rmse:11.0278
[51]	validation_0-rmse:10.4846	validation_1-rmse:11.0247
[52]	validation_0-rmse:10.4819	validation_1-rmse:11.0239
[53]	validation_0-rmse:10.4786	validation_1-rmse:11.0209
[54]	validation_0-rmse:10.4739	validation_1-rmse:11.0167
[55]	validation_0-rmse:10.4666	validation_1-rmse:11.0154
[56]	validation_0-rmse:10.4581	validation_1-rmse:11.0168
[57]	validation_0-rmse:10.4528	validation_1-rmse:11.016
[58]	validation_0-rmse:10.4508	validation_1-rmse:11.0144
[59]	validation_0-rmse:10.4484	validation_1-rmse:11.0134
[60]	validation_0-rmse:10.4393	validation_1-rmse:11.0078
[61]	validation_0-rmse:10.4316	validation_1-rmse:11.0057
[62]	validation_0-rmse:10.4287	validation_1-rmse:11.0029
[63]	validation_0-rmse:10.4264	validation_1-rmse:11.001
[64]	validation_0-rmse:10.4211	validation_1-rmse:11.0024
[65]	validation_0-rmse:10.4184	validation_1-rmse:11.0006
[66]	validation_0-rmse:10.4135	validation_1-rmse:10.9971
[67]	validation_0-rmse:10.4089	validation_1-rmse:10.9945
[68]	validation_0-rmse:10.4023	validation_1-rmse:10.9892
[69]	validation_0-rmse:10.3947	validation_1-rmse:10.9893
[70]	validation_0-rmse:10.3896	validation_1-rmse:10.9894
[71]	validation_0-rmse:10.3861	validation_1-rmse:10.9868
[72]	validation_0-rmse:10.3809	validation_1-rmse:10.9882
[73]	validation_0-rmse:10.3775	validation_1-rmse:10.9861
[74]	validation_0-rmse:10.3697	validation_1-rmse:10.9859
[75]	validation_0-rmse:10.3668	validation_1-rmse:10.986
[76]	validation_0-rmse:10.3608	validation_1-rmse:10.9832
[77]	validation_0-rmse:10.3585	validation_1-rmse:10.9816
[78]	validation_0-rmse:10.3545	validation_1-rmse:10.9791
[79]	validation_0-rmse:10.351	validation_1-rmse:10.9776
[80]	validation_0-rmse:10.3485	validation_1-rmse:10.9762
[81]	validation_0-rmse:10.3428	validation_1-rmse:10.9791
[82]	validation_0-rmse:10.3336	validation_1-rmse:10.9807
[83]	validation_0-rmse:10.3287	validation_1-rmse:10.9788
[84]	validation_0-rmse:10.3286	validation_1-rmse:10.9782
[85]	validation_0-rmse:10.3234	validation_1-rmse:10.982
[86]	validation_0-rmse:10.3216	validation_1-rmse:10.9812
[87]	validation_0-rmse:10.3207	validation_1-rmse:10.9809
[88]	validation_0-rmse:10.3192	validation_1-rmse:10.9789
[89]	validation_0-rmse:10.3155	validation_1-rmse:10.9793
[90]	validation_0-rmse:10.3114	validation_1-rmse:10.9763
[91]	validation_0-rmse:10.3106	validation_1-rmse:10.9768
[92]	validation_0-rmse:10.3094	validation_1-rmse:10.9757
[93]	validation_0-rmse:10.3061	validation_1-rmse:10.9756
[94]	validation_0-rmse:10.3032	validation_1-rmse:10.9761
[95]	validation_0-rmse:10.3001	validation_1-rmse:10.9746
[96]	validation_0-rmse:10.298	validation_1-rmse:10.9754
[97]	validation_0-rmse:10.2968	validation_1-rmse:10.9753
[98]	validation_0-rmse:10.294	validation_1-rmse:10.9731
[99]	validation_0-rmse:10.2923	validation_1-rmse:10.9722
[100]	validation_0-rmse:10.2904	validation_1-rmse:10.9709
[101]	validation_0-rmse:10.2885	validation_1-rmse:10.9709
[102]	validation_0-rmse:10.2825	validation_1-rmse:10.9711
[103]	validation_0-rmse:10.2814	validation_1-rmse:10.9706
[104]	validation_0-rmse:10.28	validation_1-rmse:10.9702
[105]	validation_0-rmse:10.2796	validation_1-rmse:10.9703
[106]	validation_0-rmse:10.2772	validation_1-rmse:10.9739
[107]	validation_0-rmse:10.2746	validation_1-rmse:10.9748
[108]	validation_0-rmse:10.2743	validation_1-rmse:10.9746
[109]	validation_0-rmse:10.2743	validation_1-rmse:10.9742
[110]	validation_0-rmse:10.2725	validation_1-rmse:10.9735
[111]	validation_0-rmse:10.2682	validation_1-rmse:10.9744
[112]	validation_0-rmse:10.2615	validation_1-rmse:10.9775
[113]	validation_0-rmse:10.2553	validation_1-rmse:10.9811
[114]	validation_0-rmse:10.2532	validation_1-rmse:10.9806
[115]	validation_0-rmse:10.2511	validation_1-rmse:10.9805
[116]	validation_0-rmse:10.2473	validation_1-rmse:10.9792
[117]	validation_0-rmse:10.2411	validation_1-rmse:10.985
[118]	validation_0-rmse:10.2346	validation_1-rmse:10.9922
[119]	validation_0-rmse:10.2335	validation_1-rmse:10.9915
[120]	validation_0-rmse:10.2306	validation_1-rmse:10.9914
[121]	validation_0-rmse:10.2292	validation_1-rmse:10.9904
[122]	validation_0-rmse:10.2288	validation_1-rmse:10.9905
[123]	validation_0-rmse:10.2248	validation_1-rmse:10.9908
[124]	validation_0-rmse:10.2233	validation_1-rmse:10.9909
[125]	validation_0-rmse:10.222	validation_1-rmse:10.9902
[126]	validation_0-rmse:10.2193	validation_1-rmse:10.9927
[127]	validation_0-rmse:10.2148	validation_1-rmse:11.0002
[128]	validation_0-rmse:10.2109	validation_1-rmse:10.998
[129]	validation_0-rmse:10.2053	validation_1-rmse:10.9992
[130]	validation_0-rmse:10.204	validation_1-rmse:10.9993
[131]	validation_0-rmse:10.1997	validation_1-rmse:11.0037
[132]	validation_0-rmse:10.1963	validation_1-rmse:11.0049
[133]	validation_0-rmse:10.1905	validation_1-rmse:11.0134
[134]	validation_0-rmse:10.1876	validation_1-rmse:11.0147
[135]	validation_0-rmse:10.1832	validation_1-rmse:11.0118
[136]	validation_0-rmse:10.182	validation_1-rmse:11.0118
[137]	validation_0-rmse:10.1786	validation_1-rmse:11.0112
[138]	validation_0-rmse:10.1728	validation_1-rmse:11.0097
[139]	validation_0-rmse:10.1716	validation_1-rmse:11.0092
[140]	validation_0-rmse:10.1663	validation_1-rmse:11.0092
[141]	validation_0-rmse:10.1649	validation_1-rmse:11.0084
[142]	validation_0-rmse:10.1627	validation_1-rmse:11.0105
[143]	validation_0-rmse:10.1589	validation_1-rmse:11.0162
[144]	validation_0-rmse:10.1564	validation_1-rmse:11.0153
[145]	validation_0-rmse:10.1516	validation_1-rmse:11.0212
[146]	validation_0-rmse:10.1477	validation_1-rmse:11.0212
[147]	validation_0-rmse:10.1471	validation_1-rmse:11.0211
[148]	validation_0-rmse:10.1464	validation_1-rmse:11.0207
[149]	validation_0-rmse:10.1433	validation_1-rmse:11.0169
[150]	validation_0-rmse:10.1418	validation_1-rmse:11.0217
[151]	validation_0-rmse:10.1417	validation_1-rmse:11.0194
[152]	validation_0-rmse:10.1403	validation_1-rmse:11.0186
[153]	validation_0-rmse:10.1398	validation_1-rmse:11.0184
[154]	validation_0-rmse:10.1368	validation_1-rmse:11.0157
Stopping. Best iteration:
[104]	validation_0-rmse:10.28	validation_1-rmse:10.9702

[0]	validation_0-rmse:24.4351	validation_1-rmse:24.7535
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.4944	validation_1-rmse:22.8221
[2]	validation_0-rmse:20.7818	validation_1-rmse:21.1231
[3]	validation_0-rmse:19.3949	validation_1-rmse:19.748
[4]	validation_0-rmse:18.2073	validation_1-rmse:18.5734
[5]	validation_0-rmse:17.1761	validation_1-rmse:17.5566
[6]	validation_0-rmse:16.1463	validation_1-rmse:16.5417
[7]	validation_0-rmse:15.3804	validation_1-rmse:15.7921
[8]	validation_0-rmse:14.6157	validation_1-rmse:15.0415
[9]	validation_0-rmse:13.9531	validation_1-rmse:14.3958
[10]	validation_0-rmse:13.3766	validation_1-rmse:13.8437
[11]	validation_0-rmse:12.8976	validation_1-rmse:13.3787
[12]	validation_0-rmse:12.4862	validation_1-rmse:12.9878
[13]	validation_0-rmse:12.2248	validation_1-rmse:12.7395
[14]	validation_0-rmse:11.9637	validation_1-rmse:12.49
[15]	validation_0-rmse:11.7082	validation_1-rmse:12.2484
[16]	validation_0-rmse:11.5193	validation_1-rmse:12.0743
[17]	validation_0-rmse:11.3308	validation_1-rmse:11.8971
[18]	validation_0-rmse:11.1945	validation_1-rmse:11.774
[19]	validation_0-rmse:11.0536	validation_1-rmse:11.6418
[20]	validation_0-rmse:10.9663	validation_1-rmse:11.562
[21]	validation_0-rmse:10.8818	validation_1-rmse:11.4878
[22]	validation_0-rmse:10.8285	validation_1-rmse:11.4405
[23]	validation_0-rmse:10.7456	validation_1-rmse:11.3662
[24]	validation_0-rmse:10.6962	validation_1-rmse:11.3233
[25]	validation_0-rmse:10.621	validation_1-rmse:11.2673
[26]	validation_0-rmse:10.5799	validation_1-rmse:11.2351
[27]	validation_0-rmse:10.5304	validation_1-rmse:11.1937
[28]	validation_0-rmse:10.4882	validation_1-rmse:11.1674
[29]	validation_0-rmse:10.4484	validation_1-rmse:11.1358
[30]	validation_0-rmse:10.4106	validation_1-rmse:11.1076
[31]	validation_0-rmse:10.3795	validation_1-rmse:11.0849
[32]	validation_0-rmse:10.362	validation_1-rmse:11.073
[33]	validation_0-rmse:10.3353	validation_1-rmse:11.0544
[34]	validation_0-rmse:10.3176	validation_1-rmse:11.0445
[35]	validation_0-rmse:10.3065	validation_1-rmse:11.0371
[36]	validation_0-rmse:10.2904	validation_1-rmse:11.0268
[37]	validation_0-rmse:10.275	validation_1-rmse:11.0157
[38]	validation_0-rmse:10.2453	validation_1-rmse:11.0027
[39]	validation_0-rmse:10.2309	validation_1-rmse:10.9941
[40]	validation_0-rmse:10.2113	validation_1-rmse:10.9874
[41]	validation_0-rmse:10.2006	validation_1-rmse:10.9833
[42]	validation_0-rmse:10.1898	validation_1-rmse:10.9785
[43]	validation_0-rmse:10.1789	validation_1-rmse:10.9716
[44]	validation_0-rmse:10.1619	validation_1-rmse:10.9669
[45]	validation_0-rmse:10.1539	validation_1-rmse:10.9663
[46]	validation_0-rmse:10.1436	validation_1-rmse:10.962
[47]	validation_0-rmse:10.1225	validation_1-rmse:10.9537
[48]	validation_0-rmse:10.1105	validation_1-rmse:10.949
[49]	validation_0-rmse:10.0981	validation_1-rmse:10.9411
[50]	validation_0-rmse:10.0887	validation_1-rmse:10.9431
[51]	validation_0-rmse:10.0784	validation_1-rmse:10.9364
[52]	validation_0-rmse:10.0704	validation_1-rmse:10.9306
[53]	validation_0-rmse:10.0626	validation_1-rmse:10.9257
[54]	validation_0-rmse:10.0511	validation_1-rmse:10.9253
[55]	validation_0-rmse:10.0415	validation_1-rmse:10.9204
[56]	validation_0-rmse:10.0345	validation_1-rmse:10.9167
[57]	validation_0-rmse:10.0302	validation_1-rmse:10.9144
[58]	validation_0-rmse:10.0241	validation_1-rmse:10.9119
[59]	validation_0-rmse:10.0182	validation_1-rmse:10.9089
[60]	validation_0-rmse:10.0113	validation_1-rmse:10.9081
[61]	validation_0-rmse:10.0009	validation_1-rmse:10.9054
[62]	validation_0-rmse:9.99868	validation_1-rmse:10.9047
[63]	validation_0-rmse:9.99165	validation_1-rmse:10.9021
[64]	validation_0-rmse:9.98613	validation_1-rmse:10.9
[65]	validation_0-rmse:9.9815	validation_1-rmse:10.8984
[66]	validation_0-rmse:9.97533	validation_1-rmse:10.8995
[67]	validation_0-rmse:9.97035	validation_1-rmse:10.8976
[68]	validation_0-rmse:9.96532	validation_1-rmse:10.8927
[69]	validation_0-rmse:9.96177	validation_1-rmse:10.8955
[70]	validation_0-rmse:9.94713	validation_1-rmse:10.8926
[71]	validation_0-rmse:9.93083	validation_1-rmse:10.8924
[72]	validation_0-rmse:9.91927	validation_1-rmse:10.8899
[73]	validation_0-rmse:9.90767	validation_1-rmse:10.8911
[74]	validation_0-rmse:9.89747	validation_1-rmse:10.8851
[75]	validation_0-rmse:9.8921	validation_1-rmse:10.8878
[76]	validation_0-rmse:9.87618	validation_1-rmse:10.8819
[77]	validation_0-rmse:9.86894	validation_1-rmse:10.8791
[78]	validation_0-rmse:9.85983	validation_1-rmse:10.8766
[79]	validation_0-rmse:9.85235	validation_1-rmse:10.8731
[80]	validation_0-rmse:9.84807	validation_1-rmse:10.8731
[81]	validation_0-rmse:9.84116	validation_1-rmse:10.8695
[82]	validation_0-rmse:9.83596	validation_1-rmse:10.8699
[83]	validation_0-rmse:9.82799	validation_1-rmse:10.8692
[84]	validation_0-rmse:9.82347	validation_1-rmse:10.868
[85]	validation_0-rmse:9.81796	validation_1-rmse:10.8647
[86]	validation_0-rmse:9.81112	validation_1-rmse:10.8623
[87]	validation_0-rmse:9.80473	validation_1-rmse:10.8602
[88]	validation_0-rmse:9.79969	validation_1-rmse:10.8627
[89]	validation_0-rmse:9.78766	validation_1-rmse:10.8601
[90]	validation_0-rmse:9.78416	validation_1-rmse:10.8598
[91]	validation_0-rmse:9.78029	validation_1-rmse:10.8606
[92]	validation_0-rmse:9.77795	validation_1-rmse:10.8599
[93]	validation_0-rmse:9.77146	validation_1-rmse:10.8628
[94]	validation_0-rmse:9.76836	validation_1-rmse:10.8644
[95]	validation_0-rmse:9.76117	validation_1-rmse:10.8639
[96]	validation_0-rmse:9.75846	validation_1-rmse:10.8633
[97]	validation_0-rmse:9.75489	validation_1-rmse:10.8622
[98]	validation_0-rmse:9.74933	validation_1-rmse:10.8607
[99]	validation_0-rmse:9.74789	validation_1-rmse:10.86
[100]	validation_0-rmse:9.74589	validation_1-rmse:10.8602
[101]	validation_0-rmse:9.74378	validation_1-rmse:10.8607
[102]	validation_0-rmse:9.74355	validation_1-rmse:10.8607
[103]	validation_0-rmse:9.73886	validation_1-rmse:10.8575
[104]	validation_0-rmse:9.72883	validation_1-rmse:10.8541
[105]	validation_0-rmse:9.72603	validation_1-rmse:10.8537
[106]	validation_0-rmse:9.71619	validation_1-rmse:10.8486
[107]	validation_0-rmse:9.70702	validation_1-rmse:10.8465
[108]	validation_0-rmse:9.70442	validation_1-rmse:10.8458
[109]	validation_0-rmse:9.70084	validation_1-rmse:10.8497
[110]	validation_0-rmse:9.69406	validation_1-rmse:10.8484
[111]	validation_0-rmse:9.68236	validation_1-rmse:10.8443
[112]	validation_0-rmse:9.67791	validation_1-rmse:10.8424
[113]	validation_0-rmse:9.67091	validation_1-rmse:10.8399
[114]	validation_0-rmse:9.6598	validation_1-rmse:10.8369
[115]	validation_0-rmse:9.65569	validation_1-rmse:10.8376
[116]	validation_0-rmse:9.6513	validation_1-rmse:10.8358
[117]	validation_0-rmse:9.64995	validation_1-rmse:10.836
[118]	validation_0-rmse:9.64746	validation_1-rmse:10.8358
[119]	validation_0-rmse:9.64489	validation_1-rmse:10.8339
[120]	validation_0-rmse:9.644	validation_1-rmse:10.8339
[121]	validation_0-rmse:9.64283	validation_1-rmse:10.8335
[122]	validation_0-rmse:9.63309	validation_1-rmse:10.8318
[123]	validation_0-rmse:9.63043	validation_1-rmse:10.8316
[124]	validation_0-rmse:9.6295	validation_1-rmse:10.8311
[125]	validation_0-rmse:9.62498	validation_1-rmse:10.8307
[126]	validation_0-rmse:9.62341	validation_1-rmse:10.8319
[127]	validation_0-rmse:9.62167	validation_1-rmse:10.8324
[128]	validation_0-rmse:9.61978	validation_1-rmse:10.8325
[129]	validation_0-rmse:9.61253	validation_1-rmse:10.8372
[130]	validation_0-rmse:9.61084	validation_1-rmse:10.8377
[131]	validation_0-rmse:9.60623	validation_1-rmse:10.8391
[132]	validation_0-rmse:9.60573	validation_1-rmse:10.8399
[133]	validation_0-rmse:9.59817	validation_1-rmse:10.847
[134]	validation_0-rmse:9.59621	validation_1-rmse:10.8479
[135]	validation_0-rmse:9.59382	validation_1-rmse:10.8474
[136]	validation_0-rmse:9.59126	validation_1-rmse:10.8471
[137]	validation_0-rmse:9.58734	validation_1-rmse:10.8465
[138]	validation_0-rmse:9.58316	validation_1-rmse:10.8445
[139]	validation_0-rmse:9.57999	validation_1-rmse:10.8482
[140]	validation_0-rmse:9.577	validation_1-rmse:10.8473
[141]	validation_0-rmse:9.57482	validation_1-rmse:10.8473
[142]	validation_0-rmse:9.56924	validation_1-rmse:10.8464
[143]	validation_0-rmse:9.56827	validation_1-rmse:10.846
[144]	validation_0-rmse:9.56261	validation_1-rmse:10.8474
[145]	validation_0-rmse:9.56172	validation_1-rmse:10.8484
[146]	validation_0-rmse:9.55566	validation_1-rmse:10.8461
[147]	validation_0-rmse:9.55121	validation_1-rmse:10.8453
[148]	validation_0-rmse:9.55019	validation_1-rmse:10.8452
[149]	validation_0-rmse:9.55	validation_1-rmse:10.8452
[150]	validation_0-rmse:9.54558	validation_1-rmse:10.8479
[151]	validation_0-rmse:9.54594	validation_1-rmse:10.8477
[152]	validation_0-rmse:9.53996	validation_1-rmse:10.8463
[153]	validation_0-rmse:9.53717	validation_1-rmse:10.846
[154]	validation_0-rmse:9.53509	validation_1-rmse:10.8434
[155]	validation_0-rmse:9.53033	validation_1-rmse:10.8411
[156]	validation_0-rmse:9.52538	validation_1-rmse:10.84
[157]	validation_0-rmse:9.52407	validation_1-rmse:10.8402
[158]	validation_0-rmse:9.51655	validation_1-rmse:10.8356
[159]	validation_0-rmse:9.51577	validation_1-rmse:10.8359
[160]	validation_0-rmse:9.51079	validation_1-rmse:10.8333
[161]	validation_0-rmse:9.50997	validation_1-rmse:10.8339
[162]	validation_0-rmse:9.50746	validation_1-rmse:10.8338
[163]	validation_0-rmse:9.50386	validation_1-rmse:10.8321
[164]	validation_0-rmse:9.49906	validation_1-rmse:10.8329
[165]	validation_0-rmse:9.49453	validation_1-rmse:10.8335
[166]	validation_0-rmse:9.49414	validation_1-rmse:10.834
[167]	validation_0-rmse:9.49329	validation_1-rmse:10.8346
[168]	validation_0-rmse:9.48604	validation_1-rmse:10.8302
[169]	validation_0-rmse:9.48553	validation_1-rmse:10.8294
[170]	validation_0-rmse:9.48418	validation_1-rmse:10.829
[171]	validation_0-rmse:9.47833	validation_1-rmse:10.8289
[172]	validation_0-rmse:9.47674	validation_1-rmse:10.8294
[173]	validation_0-rmse:9.46866	validation_1-rmse:10.8287
[174]	validation_0-rmse:9.46691	validation_1-rmse:10.8284
[175]	validation_0-rmse:9.46485	validation_1-rmse:10.8296
[176]	validation_0-rmse:9.45777	validation_1-rmse:10.8264
[177]	validation_0-rmse:9.45769	validation_1-rmse:10.8264
[178]	validation_0-rmse:9.4505	validation_1-rmse:10.8339
[179]	validation_0-rmse:9.44967	validation_1-rmse:10.8346
[180]	validation_0-rmse:9.446	validation_1-rmse:10.8405
[181]	validation_0-rmse:9.44381	validation_1-rmse:10.8405
[182]	validation_0-rmse:9.44208	validation_1-rmse:10.8406
[183]	validation_0-rmse:9.44072	validation_1-rmse:10.8406
[184]	validation_0-rmse:9.44047	validation_1-rmse:10.8406
[185]	validation_0-rmse:9.44016	validation_1-rmse:10.8417
[186]	validation_0-rmse:9.43961	validation_1-rmse:10.8414
[187]	validation_0-rmse:9.43953	validation_1-rmse:10.8414
[188]	validation_0-rmse:9.43542	validation_1-rmse:10.8404
[189]	validation_0-rmse:9.43515	validation_1-rmse:10.8392
[190]	validation_0-rmse:9.43506	validation_1-rmse:10.8391
[191]	validation_0-rmse:9.43128	validation_1-rmse:10.8382
[192]	validation_0-rmse:9.4312	validation_1-rmse:10.8382
[193]	validation_0-rmse:9.41997	validation_1-rmse:10.839
[194]	validation_0-rmse:9.41541	validation_1-rmse:10.8401
[195]	validation_0-rmse:9.41535	validation_1-rmse:10.84
[196]	validation_0-rmse:9.41057	validation_1-rmse:10.8444
[197]	validation_0-rmse:9.40587	validation_1-rmse:10.8423
[198]	validation_0-rmse:9.40282	validation_1-rmse:10.8424
[199]	validation_0-rmse:9.40192	validation_1-rmse:10.8426
[200]	validation_0-rmse:9.39617	validation_1-rmse:10.8404
[201]	validation_0-rmse:9.39015	validation_1-rmse:10.8456
[202]	validation_0-rmse:9.38911	validation_1-rmse:10.8455
[203]	validation_0-rmse:9.38778	validation_1-rmse:10.8467
[204]	validation_0-rmse:9.38266	validation_1-rmse:10.8455
[205]	validation_0-rmse:9.38157	validation_1-rmse:10.8454
[206]	validation_0-rmse:9.38131	validation_1-rmse:10.8458
[207]	validation_0-rmse:9.37539	validation_1-rmse:10.8494
[208]	validation_0-rmse:9.37457	validation_1-rmse:10.8489
[209]	validation_0-rmse:9.37347	validation_1-rmse:10.8491
[210]	validation_0-rmse:9.36519	validation_1-rmse:10.8457
[211]	validation_0-rmse:9.36127	validation_1-rmse:10.8441
[212]	validation_0-rmse:9.36076	validation_1-rmse:10.844
[213]	validation_0-rmse:9.35909	validation_1-rmse:10.8441
[214]	validation_0-rmse:9.35858	validation_1-rmse:10.8448
[215]	validation_0-rmse:9.35615	validation_1-rmse:10.8428
[216]	validation_0-rmse:9.35407	validation_1-rmse:10.8446
[217]	validation_0-rmse:9.3523	validation_1-rmse:10.8445
[218]	validation_0-rmse:9.34916	validation_1-rmse:10.8437
[219]	validation_0-rmse:9.34868	validation_1-rmse:10.8444
[220]	validation_0-rmse:9.34865	validation_1-rmse:10.8444
[221]	validation_0-rmse:9.3464	validation_1-rmse:10.8449
[222]	validation_0-rmse:9.3409	validation_1-rmse:10.8433
[223]	validation_0-rmse:9.33878	validation_1-rmse:10.8449
[224]	validation_0-rmse:9.33849	validation_1-rmse:10.8421
[225]	validation_0-rmse:9.33661	validation_1-rmse:10.8408
[226]	validation_0-rmse:9.3366	validation_1-rmse:10.8408
Stopping. Best iteration:
[176]	validation_0-rmse:9.45777	validation_1-rmse:10.8264

{'colsample_bytree': 0.75, 'gamma': 2.8000000000000003, 'learning_rate': 0, 'max_depth': 3, 'min_child_weight': 3.0, 'n_estimators': 0, 'nthread': 0, 'objective': 0, 'reg_alpha': 3.2, 'reg_lambda': 0.7000000000000001, 'subsample': 1.0}
dict_keys(['colsample_bytree', 'gamma', 'learning_rate', 'max_depth', 'min_child_weight', 'n_estimators', 'nthread', 'objective', 'reg_alpha', 'reg_lambda', 'subsample'])



In [74]: %paste
aaa = df_result_hyperopt[:]

df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")

## -- End pasted text --
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-74-7ce4c7afb9f7> in <module>()
----> 1 aaa = df_result_hyperopt[:]
      2
      3 df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
      4 df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")

NameError: name 'df_result_hyperopt' is not defined

In [75]: %paste
aaa = list_result_hyperopt[:]

df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")

## -- End pasted text --
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-75-c90a3dd19794> in <module>()
      1 aaa = list_result_hyperopt[:]
      2
----> 3 df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
      4 df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")

NameError: name 'space4rf' is not defined

In [76]: aaa.keys()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-76-bd86e2fccd06> in <module>()
----> 1 aaa.keys()

AttributeError: 'list' object has no attribute 'keys'

In [77]: %paste
aaa = list_result_hyperopt[:]

df_result_hyperopt = pd.DataFrame(columns=[np.append('score', list(xgbr_d.keys()))])
df_result_hyperopt.to_csv('data/hyperopt_output.csv', sep=";")

## -- End pasted text --

In [78]: cat data/hyperopt_output.csv
;score;gamma;learning_rate;colsample_bytree;max_depth;min_child_weight;subsample;nthread;n_estimators;objective;reg_lambda;reg_alpha

In [79]: aaa
Out[79]:
[(11.032122534656253,
  [0.5,
   0.6000000000000001,
   0.1,
   5,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.7,
   2.3000000000000003,
   0.6000000000000001,
   129]),
 (10.867711346768681,
  [0.75,
   1.7000000000000002,
   0.1,
   8,
   2.0,
   1500,
   -1,
   'reg:linear',
   3.1,
   3.6,
   0.75,
   263]),
 (11.056675532819281,
  [0.8,
   4.800000000000001,
   0.1,
   6,
   4.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.6,
   0.65,
   71]),
 (10.99684259948922,
  [0.4,
   1.0,
   0.1,
   5,
   4.0,
   1500,
   -1,
   'reg:linear',
   1.7000000000000002,
   3.6,
   0.9,
   131]),
 (10.789125157766302,
  [0.75,
   2.8000000000000003,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   0.7000000000000001,
   3.2,
   1.0,
   245]),
 (10.869080687503255,
  [0.55,
   1.6,
   0.1,
   7,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.8000000000000003,
   1.9000000000000001,
   0.8,
   244]),
 (11.003548402352093,
  [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]),
 (10.890873408731197,
  [0.7000000000000001,
   0.6000000000000001,
   0.1,
   9,
   3.0,
   1500,
   -1,
   'reg:linear',
   1.0,
   1.8,
   0.65,
   151]),
 (11.015703670319155,
  [0.75,
   2.9000000000000004,
   0.1,
   5,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.4000000000000004,
   2.2,
   0.9,
   104]),
 (10.840786277522257,
  [0.45,
   4.0,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.4000000000000004,
   0.8,
   176])]

In [80]: print(format('aaa'))
aaa

In [81]: print(format(aaa))
[(11.032122534656253, [0.5, 0.6000000000000001, 0.1, 5, 2.0, 1500, -1, 'reg:linear', 2.7, 2.3000000000000003, 0.6000000000000001, 129]), (10.867711346768681, [0.75, 1.7000000000000002, 0.1, 8, 2.0, 1500, -1, 'reg:linear', 3.1, 3.6, 0.75, 263]), (11.056675532819281, [0.8, 4.800000000000001, 0.1, 6, 4.0, 1500, -1, 'reg:linear', 3.0, 2.6, 0.65, 71]), (10.99684259948922, [0.4, 1.0, 0.1, 5, 4.0, 1500, -1, 'reg:linear', 1.7000000000000002, 3.6, 0.9, 131]), (10.789125157766302, [0.75, 2.8000000000000003, 0.1, 8, 3.0, 1500, -1, 'reg:linear', 0.7000000000000001, 3.2, 1.0, 245]), (10.869080687503255, [0.55, 1.6, 0.1, 7, 2.0, 1500, -1, 'reg:linear', 2.8000000000000003, 1.9000000000000001, 0.8, 244]), (11.003548402352093, [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]), (10.890873408731197, [0.7000000000000001, 0.6000000000000001, 0.1, 9, 3.0, 1500, -1, 'reg:linear', 1.0, 1.8, 0.65, 151]), (11.015703670319155, [0.75, 2.9000000000000004, 0.1, 5, 3.0, 1500, -1, 'reg:linear', 3.4000000000000004, 2.2, 0.9, 104]), (10.840786277522257, [0.45, 4.0, 0.1, 8, 3.0, 1500, -1, 'reg:linear', 3.0, 2.4000000000000004, 0.8, 176])]

In [82]: print(format(aaa))
[(11.032122534656253, [0.5, 0.6000000000000001, 0.1, 5, 2.0, 1500, -1, 'reg:linear', 2.7, 2.3000000000000003, 0.6000000000000001, 129]), (10.867711346768681, [0.75, 1.7000000000000002, 0.1, 8, 2.0, 1500, -1, 'reg:linear', 3.1, 3.6, 0.75, 263]), (11.056675532819281, [0.8, 4.800000000000001, 0.1, 6, 4.0, 1500, -1, 'reg:linear', 3.0, 2.6, 0.65, 71]), (10.99684259948922, [0.4, 1.0, 0.1, 5, 4.0, 1500, -1, 'reg:linear', 1.7000000000000002, 3.6, 0.9, 131]), (10.789125157766302, [0.75, 2.8000000000000003, 0.1, 8, 3.0, 1500, -1, 'reg:linear', 0.7000000000000001, 3.2, 1.0, 245]), (10.869080687503255, [0.55, 1.6, 0.1, 7, 2.0, 1500, -1, 'reg:linear', 2.8000000000000003, 1.9000000000000001, 0.8, 244]), (11.003548402352093, [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]), (10.890873408731197, [0.7000000000000001, 0.6000000000000001, 0.1, 9, 3.0, 1500, -1, 'reg:linear', 1.0, 1.8, 0.65, 151]), (11.015703670319155, [0.75, 2.9000000000000004, 0.1, 5, 3.0, 1500, -1, 'reg:linear', 3.4000000000000004, 2.2, 0.9, 104]), (10.840786277522257, [0.45, 4.0, 0.1, 8, 3.0, 1500, -1, 'reg:linear', 3.0, 2.4000000000000004, 0.8, 176])]

In [83]:

In [83]:

In [83]: aaa
Out[83]:
[(11.032122534656253,
  [0.5,
   0.6000000000000001,
   0.1,
   5,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.7,
   2.3000000000000003,
   0.6000000000000001,
   129]),
 (10.867711346768681,
  [0.75,
   1.7000000000000002,
   0.1,
   8,
   2.0,
   1500,
   -1,
   'reg:linear',
   3.1,
   3.6,
   0.75,
   263]),
 (11.056675532819281,
  [0.8,
   4.800000000000001,
   0.1,
   6,
   4.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.6,
   0.65,
   71]),
 (10.99684259948922,
  [0.4,
   1.0,
   0.1,
   5,
   4.0,
   1500,
   -1,
   'reg:linear',
   1.7000000000000002,
   3.6,
   0.9,
   131]),
 (10.789125157766302,
  [0.75,
   2.8000000000000003,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   0.7000000000000001,
   3.2,
   1.0,
   245]),
 (10.869080687503255,
  [0.55,
   1.6,
   0.1,
   7,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.8000000000000003,
   1.9000000000000001,
   0.8,
   244]),
 (11.003548402352093,
  [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]),
 (10.890873408731197,
  [0.7000000000000001,
   0.6000000000000001,
   0.1,
   9,
   3.0,
   1500,
   -1,
   'reg:linear',
   1.0,
   1.8,
   0.65,
   151]),
 (11.015703670319155,
  [0.75,
   2.9000000000000004,
   0.1,
   5,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.4000000000000004,
   2.2,
   0.9,
   104]),
 (10.840786277522257,
  [0.45,
   4.0,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.4000000000000004,
   0.8,
   176])]

In [84]: hyper_parametres
Out[84]:
{'colsample_bytree': 0.75,
 'gamma': 2.8000000000000003,
 'learning_rate': 0.1,
 'max_depth': 8,
 'min_child_weight': 3.0,
 'n_estimators': 1500,
 'nthread': -1,
 'objective': 'reg:linear',
 'reg_alpha': 3.2,
 'reg_lambda': 0.7000000000000001,
 'subsample': 1.0}

In [85]: params
Out[85]:
{'max_depth': [8, 9, 10, 11],
 'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100]}

In [86]: %paste
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

xgb_hyperopt1 = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
xgb_hyperopt1.fit(train[col], np.log1p(train['visitors'].values))
xgbpred1 = xgb_hyperopt1.predict(train[col])

## -- End pasted text --
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
~/python3/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   2441             try:
-> 2442                 return self._engine.get_loc(key)
   2443             except KeyError:

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)()

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)()

KeyError: 'visitors'

During handling of the above exception, another exception occurred:

KeyError                                  Traceback (most recent call last)
<ipython-input-86-040b0008548f> in <module>()
      3
      4 xgb_hyperopt1 = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
----> 5 xgb_hyperopt1.fit(train[col], np.log1p(train['visitors'].values))
      6 xgbpred1 = xgb_hyperopt1.predict(train[col])

~/python3/lib/python3.6/site-packages/pandas/core/frame.py in __getitem__(self, key)
   1962             return self._getitem_multilevel(key)
   1963         else:
-> 1964             return self._getitem_column(key)
   1965
   1966     def _getitem_column(self, key):

~/python3/lib/python3.6/site-packages/pandas/core/frame.py in _getitem_column(self, key)
   1969         # get column
   1970         if self.columns.is_unique:
-> 1971             return self._get_item_cache(key)
   1972
   1973         # duplicate columns & possible reduce dimensionality

~/python3/lib/python3.6/site-packages/pandas/core/generic.py in _get_item_cache(self, item)
   1643         res = cache.get(item)
   1644         if res is None:
-> 1645             values = self._data.get(item)
   1646             res = self._box_item_values(item, values)
   1647             cache[item] = res

~/python3/lib/python3.6/site-packages/pandas/core/internals.py in get(self, item, fastpath)
   3588
   3589             if not isnull(item):
-> 3590                 loc = self.items.get_loc(item)
   3591             else:
   3592                 indexer = np.arange(len(self.items))[isnull(self.items)]

~/python3/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   2442                 return self._engine.get_loc(key)
   2443             except KeyError:
-> 2444                 return self._engine.get_loc(self._maybe_cast_indexer(key))
   2445
   2446         indexer = self.get_indexer([key], method=method, tolerance=tolerance)

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)()

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)()

KeyError: 'visitors'

In [87]: %paste
xgb_hyperopt1 = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
xgb_hyperopt1.fit(X_train, y_train)
xgbpred1 = xgb_hyperopt1.predict(X_test)

## -- End pasted text --

In [88]: %paste
train_error_xgb_hyperopt1 = round(mean_squared_error(y_train, xgb_hyperopt1.predict(X_train)), 3)
test_error_xgb_hyperopt1 = round(mean_squared_error(y_test, xgb_hyperopt1.predict(X_test)), 3)
print("train error: {}".format(train_error_xgb_hyperopt))
print("test error: {}".format(test_error_xgb_hyperopt))

## -- End pasted text --
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-88-ee44d9250f67> in <module>()
      1 train_error_xgb_hyperopt1 = round(mean_squared_error(y_train, xgb_hyperopt1.predict(X_train)), 3)
      2 test_error_xgb_hyperopt1 = round(mean_squared_error(y_test, xgb_hyperopt1.predict(X_test)), 3)
----> 3 print("train error: {}".format(train_error_xgb_hyperopt))
      4 print("test error: {}".format(test_error_xgb_hyperopt))

NameError: name 'train_error_xgb_hyperopt' is not defined

In [89]: %paste
train_error_xgb_hyperopt1 = round(mean_squared_error(y_train, xgb_hyperopt1.predict(X_train)), 3)
test_error_xgb_hyperopt1 = round(mean_squared_error(y_test, xgb_hyperopt1.predict(X_test)), 3)
print("train error: {}".format(train_error_xgb_hyperopt1))
print("test error: {}".format(test_error_xgb_hyperopt1))

## -- End pasted text --
train error: 111.321
test error: 122.213

In [90]: X_test.values
Out[90]:
array([[  5.00000000e+00,   2.01700000e+03,   3.00000000e+00, ...,
          4.56934770e+00,   1.75365828e+02,   3.99000000e+02],
       [  2.00000000e+00,   2.01600000e+03,   7.00000000e+00, ...,
          3.40299540e+00,   1.79139479e+02,   5.45000000e+02],
       [  3.00000000e+00,   2.01600000e+03,   9.00000000e+00, ...,
          8.86702260e+00,   1.70140110e+02,   6.69000000e+02],
       ...,
       [  0.00000000e+00,   2.01600000e+03,   6.00000000e+00, ...,
          1.38805849e+01,   1.63982029e+02,   2.22000000e+02],
       [  1.00000000e+00,   2.01600000e+03,   1.20000000e+01, ...,
          4.50153690e+00,   1.75442512e+02,   7.70000000e+02],
       [  5.00000000e+00,   2.01600000e+03,   7.00000000e+00, ...,
          1.38805849e+01,   1.63982029e+02,   3.53000000e+02]])

In [91]: %paste
xgb_hyperopt1.fit(X_train, np.log1p(X_test.values))

## -- End pasted text --
[18:04:15] dmlc-core/include/dmlc/./logging.h:300: [18:04:15] src/objective/regression_obj.cc:44: Check failed: preds.size() == info.labels.size() (201686 vs. 50422) labels are not correctly providedpreds.size=201686, label.size=50422

Stack trace returned 6 entries:
[bt] (0) 0   libxgboost.dylib                    0x0000000109275fc8 _ZN4dmlc15LogMessageFatalD2Ev + 40
[bt] (1) 1   libxgboost.dylib                    0x00000001092e1a09 _ZN7xgboost3obj10RegLossObjINS0_16LinearSquareLossEE11GetGradientERKNSt3__16vectorIfNS4_9allocatorIfEEEERKNS_8MetaInfoEiPNS5_INS_6detail18bst_gpair_internalIfEENS6_ISG_EEEE + 601
[bt] (2) 2   libxgboost.dylib                    0x0000000109272616 _ZN7xgboost11LearnerImpl13UpdateOneIterEiPNS_7DMatrixE + 1014
[bt] (3) 3   libxgboost.dylib                    0x000000010928b4ef XGBoosterUpdateOneIter + 79
[bt] (4) 4   _ctypes.cpython-36m-darwin.so       0x00000001054b142f ffi_call_unix64 + 79
[bt] (5) 5   ???                                 0x00007ffeeceb8310 0x0 + 140732873278224

---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
<ipython-input-91-413ea2cf25fe> in <module>()
----> 1 xgb_hyperopt1.fit(X_train, np.log1p(X_test.values))

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/sklearn.py in fit(self, X, y, sample_weight, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model)
    291                               early_stopping_rounds=early_stopping_rounds,
    292                               evals_result=evals_result, obj=obj, feval=feval,
--> 293                               verbose_eval=verbose, xgb_model=xgb_model)
    294
    295         if evals_result:

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, learning_rates)
    202                            evals=evals,
    203                            obj=obj, feval=feval,
--> 204                            xgb_model=xgb_model, callbacks=callbacks)
    205
    206

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/training.py in _train_internal(params, dtrain, num_boost_round, evals, obj, feval, xgb_model, callbacks)
     72         # Skip the first update if it is a recovery step.
     73         if version % 2 == 0:
---> 74             bst.update(dtrain, i, obj)
     75             bst.save_rabit_checkpoint()
     76             version += 1

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/core.py in update(self, dtrain, iteration, fobj)
    896         if fobj is None:
    897             _check_call(_LIB.XGBoosterUpdateOneIter(self.handle, ctypes.c_int(iteration),
--> 898                                                     dtrain.handle))
    899         else:
    900             pred = self.predict(dtrain)

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/core.py in _check_call(ret)
    128     """
    129     if ret != 0:
--> 130         raise XGBoostError(_LIB.XGBGetLastError())
    131
    132

XGBoostError: b'[18:04:15] src/objective/regression_obj.cc:44: Check failed: preds.size() == info.labels.size() (201686 vs. 50422) labels are not correctly providedpreds.size=201686, label.size=50422\n\nStack trace returned 6 entries:\n[bt] (0) 0   libxgboost.dylib                    0x0000000109275fc8 _ZN4dmlc15LogMessageFatalD2Ev + 40\n[bt] (1) 1   libxgboost.dylib                    0x00000001092e1a09 _ZN7xgboost3obj10RegLossObjINS0_16LinearSquareLossEE11GetGradientERKNSt3__16vectorIfNS4_9allocatorIfEEEERKNS_8MetaInfoEiPNS5_INS_6detail18bst_gpair_internalIfEENS6_ISG_EEEE + 601\n[bt] (2) 2   libxgboost.dylib                    0x0000000109272616 _ZN7xgboost11LearnerImpl13UpdateOneIterEiPNS_7DMatrixE + 1014\n[bt] (3) 3   libxgboost.dylib                    0x000000010928b4ef XGBoosterUpdateOneIter + 79\n[bt] (4) 4   _ctypes.cpython-36m-darwin.so       0x00000001054b142f ffi_call_unix64 + 79\n[bt] (5) 5   ???                                 0x00007ffeeceb8310 0x0 + 140732873278224\n'

In [92]: %paste
xgb_hyperopt3 = XGBRegressor(learning_rate=0.1, n_estimators=1500, subsample=0.8, colsample_bytree=0.75, max_depth=1, gamma = 0.6000000000000001,
                   min_child_weight=2, objective="reg:linear", reg_alpha=0.1, reg_lambda=0.5  )
xgb_hyperopt3.fit(X_train, y_train)
xgbpred3 = xgb_hyperopt3.predict(X_test)

## -- End pasted text --

In [93]: %paste
print('RMSE XGBRegressor: ', RMSLE(np.log1p(train['visitors'].values), xgbpred3))
## -- End pasted text --
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
~/python3/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   2441             try:
-> 2442                 return self._engine.get_loc(key)
   2443             except KeyError:

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)()

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)()

KeyError: 'visitors'

During handling of the above exception, another exception occurred:

KeyError                                  Traceback (most recent call last)
<ipython-input-93-ae6ebcc8f9b0> in <module>()
----> 1 print('RMSE XGBRegressor: ', RMSLE(np.log1p(train['visitors'].values), xgbpred3))

~/python3/lib/python3.6/site-packages/pandas/core/frame.py in __getitem__(self, key)
   1962             return self._getitem_multilevel(key)
   1963         else:
-> 1964             return self._getitem_column(key)
   1965
   1966     def _getitem_column(self, key):

~/python3/lib/python3.6/site-packages/pandas/core/frame.py in _getitem_column(self, key)
   1969         # get column
   1970         if self.columns.is_unique:
-> 1971             return self._get_item_cache(key)
   1972
   1973         # duplicate columns & possible reduce dimensionality

~/python3/lib/python3.6/site-packages/pandas/core/generic.py in _get_item_cache(self, item)
   1643         res = cache.get(item)
   1644         if res is None:
-> 1645             values = self._data.get(item)
   1646             res = self._box_item_values(item, values)
   1647             cache[item] = res

~/python3/lib/python3.6/site-packages/pandas/core/internals.py in get(self, item, fastpath)
   3588
   3589             if not isnull(item):
-> 3590                 loc = self.items.get_loc(item)
   3591             else:
   3592                 indexer = np.arange(len(self.items))[isnull(self.items)]

~/python3/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   2442                 return self._engine.get_loc(key)
   2443             except KeyError:
-> 2444                 return self._engine.get_loc(self._maybe_cast_indexer(key))
   2445
   2446         indexer = self.get_indexer([key], method=method, tolerance=tolerance)

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5280)()

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas/_libs/index.c:5126)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20523)()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item (pandas/_libs/hashtable.c:20477)()

KeyError: 'visitors'

In [94]: clear

In [95]:

In [95]:

In [95]:

In [95]: aaa
Out[95]:
[(11.032122534656253,
  [0.5,
   0.6000000000000001,
   0.1,
   5,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.7,
   2.3000000000000003,
   0.6000000000000001,
   129]),
 (10.867711346768681,
  [0.75,
   1.7000000000000002,
   0.1,
   8,
   2.0,
   1500,
   -1,
   'reg:linear',
   3.1,
   3.6,
   0.75,
   263]),
 (11.056675532819281,
  [0.8,
   4.800000000000001,
   0.1,
   6,
   4.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.6,
   0.65,
   71]),
 (10.99684259948922,
  [0.4,
   1.0,
   0.1,
   5,
   4.0,
   1500,
   -1,
   'reg:linear',
   1.7000000000000002,
   3.6,
   0.9,
   131]),
 (10.789125157766302,
  [0.75,
   2.8000000000000003,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   0.7000000000000001,
   3.2,
   1.0,
   245]),
 (10.869080687503255,
  [0.55,
   1.6,
   0.1,
   7,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.8000000000000003,
   1.9000000000000001,
   0.8,
   244]),
 (11.003548402352093,
  [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]),
 (10.890873408731197,
  [0.7000000000000001,
   0.6000000000000001,
   0.1,
   9,
   3.0,
   1500,
   -1,
   'reg:linear',
   1.0,
   1.8,
   0.65,
   151]),
 (11.015703670319155,
  [0.75,
   2.9000000000000004,
   0.1,
   5,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.4000000000000004,
   2.2,
   0.9,
   104]),
 (10.840786277522257,
  [0.45,
   4.0,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.4000000000000004,
   0.8,
   176])]

In [96]: list_result_hyperopt
Out[96]:
[(11.032122534656253,
  [0.5,
   0.6000000000000001,
   0.1,
   5,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.7,
   2.3000000000000003,
   0.6000000000000001,
   129]),
 (10.867711346768681,
  [0.75,
   1.7000000000000002,
   0.1,
   8,
   2.0,
   1500,
   -1,
   'reg:linear',
   3.1,
   3.6,
   0.75,
   263]),
 (11.056675532819281,
  [0.8,
   4.800000000000001,
   0.1,
   6,
   4.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.6,
   0.65,
   71]),
 (10.99684259948922,
  [0.4,
   1.0,
   0.1,
   5,
   4.0,
   1500,
   -1,
   'reg:linear',
   1.7000000000000002,
   3.6,
   0.9,
   131]),
 (10.789125157766302,
  [0.75,
   2.8000000000000003,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   0.7000000000000001,
   3.2,
   1.0,
   245]),
 (10.869080687503255,
  [0.55,
   1.6,
   0.1,
   7,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.8000000000000003,
   1.9000000000000001,
   0.8,
   244]),
 (11.003548402352093,
  [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]),
 (10.890873408731197,
  [0.7000000000000001,
   0.6000000000000001,
   0.1,
   9,
   3.0,
   1500,
   -1,
   'reg:linear',
   1.0,
   1.8,
   0.65,
   151]),
 (11.015703670319155,
  [0.75,
   2.9000000000000004,
   0.1,
   5,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.4000000000000004,
   2.2,
   0.9,
   104]),
 (10.840786277522257,
  [0.45,
   4.0,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.4000000000000004,
   0.8,
   176])]

In [97]: list_result_hyperopt[0]
Out[97]:
(11.032122534656253,
 [0.5,
  0.6000000000000001,
  0.1,
  5,
  2.0,
  1500,
  -1,
  'reg:linear',
  2.7,
  2.3000000000000003,
  0.6000000000000001,
  129])

In [98]: s = sorted(list_result_hyperopt)

In [99]: s
Out[99]:
[(10.789125157766302,
  [0.75,
   2.8000000000000003,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   0.7000000000000001,
   3.2,
   1.0,
   245]),
 (10.840786277522257,
  [0.45,
   4.0,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.4000000000000004,
   0.8,
   176]),
 (10.867711346768681,
  [0.75,
   1.7000000000000002,
   0.1,
   8,
   2.0,
   1500,
   -1,
   'reg:linear',
   3.1,
   3.6,
   0.75,
   263]),
 (10.869080687503255,
  [0.55,
   1.6,
   0.1,
   7,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.8000000000000003,
   1.9000000000000001,
   0.8,
   244]),
 (10.890873408731197,
  [0.7000000000000001,
   0.6000000000000001,
   0.1,
   9,
   3.0,
   1500,
   -1,
   'reg:linear',
   1.0,
   1.8,
   0.65,
   151]),
 (10.99684259948922,
  [0.4,
   1.0,
   0.1,
   5,
   4.0,
   1500,
   -1,
   'reg:linear',
   1.7000000000000002,
   3.6,
   0.9,
   131]),
 (11.003548402352093,
  [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]),
 (11.015703670319155,
  [0.75,
   2.9000000000000004,
   0.1,
   5,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.4000000000000004,
   2.2,
   0.9,
   104]),
 (11.032122534656253,
  [0.5,
   0.6000000000000001,
   0.1,
   5,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.7,
   2.3000000000000003,
   0.6000000000000001,
   129]),
 (11.056675532819281,
  [0.8,
   4.800000000000001,
   0.1,
   6,
   4.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.6,
   0.65,
   71])]

In [100]: zip(s)
Out[100]: <zip at 0x116764e88>

In [101]: list(zip(s))[0]
Out[101]:
((10.789125157766302,
  [0.75,
   2.8000000000000003,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   0.7000000000000001,
   3.2,
   1.0,
   245]),)

In [102]: list(zip(*s))[0]
Out[102]:
(10.789125157766302,
 10.840786277522257,
 10.867711346768681,
 10.869080687503255,
 10.890873408731197,
 10.99684259948922,
 11.003548402352093,
 11.015703670319155,
 11.032122534656253,
 11.056675532819281)

In [103]: s[0]
Out[103]:
(10.789125157766302,
 [0.75,
  2.8000000000000003,
  0.1,
  8,
  3.0,
  1500,
  -1,
  'reg:linear',
  0.7000000000000001,
  3.2,
  1.0,
  245])

In [104]: clear

In [105]: %paste
def get_final_parameters(best_params, origin_model_d, current_model):
    """
    Fix the string parameters returned by hyperopt.
    Use the index given by hyperopt to find the real string value
    for a specific parameter
    """
    for element_p in list(best_params.keys()):
        if isinstance(
                origin_model_d[current_model][1][element_p],
                hyperopt.pyll.base.Apply):
            if origin_model_d[current_model][1][element_p].name == "float":
                pass
            elif origin_model_d[current_model][1][element_p].name == "switch":
                apply_obj = origin_model_d[current_model][1][element_p]
                literal_obj = apply_obj.pos_args[best_params[element_p] + 1]
                best_params[element_p] = literal_obj.obj
            else:
                pass
        else:
            pass
    return best_params

def regression_params_opt(
        origin_model_d,
        current_model,
        X_train,
        X_test,
        y_train,
        y_test):
    print(current_model)
    best = 0
    max_eval = 10
    trials = Trials()

    def rmse_score(params):
        model_fit = origin_model_d[current_model][0](
            **params).fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(X_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        params['best_iteration'] = model_fit.best_iteration
        list_result_hyperopt.append((loss, params))
        return {'loss': loss, 'status': STATUS_OK}

    best = fmin(rmse_score,
                origin_model_d[current_model][1],
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials)
    print(best)
    print(best.keys())
    print("\n")
    return get_final_parameters(best, origin_model_d, current_model)

if __name__ == '__main__':

    xgbr_d = {'gamma': hp.quniform('gamma', 0.0, 5.0, 0.1),
              'learning_rate': hp.choice('learning_rate', [0.1]),
              'colsample_bytree': hp.quniform('colsample_bytree',
                                              0.3,
                                              1.,
                                              0.05),
              'max_depth': hp.choice('max_depth', list(range(5, 10))),
              'min_child_weight': hp.quniform('min_child_weight', 1., 5., 1),
              'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
              'nthread': hp.choice('nthread', [-1]),
              'n_estimators': hp.choice('n_estimators', [1500]),
              'objective': hp.choice('objective', ['reg:linear']),
              'reg_lambda': hp.quniform('reg_alpha', 0.0, 4.0, 0.1),
              'reg_alpha': hp.quniform('reg_lambda', 0.0, 4.0, 0.1)}

    base_model = {"XGBRegressor": [XGBRegressor, xgbr_d]}

    list_result_hyperopt = []
    hyper_parametres = regression_params_opt(
        origin_model_d=base_model,
        current_model="XGBRegressor",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

## -- End pasted text --
XGBRegressor
[0]	validation_0-rmse:24.239	validation_1-rmse:24.5526
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3309	validation_1-rmse:22.6492
[2]	validation_0-rmse:20.653	validation_1-rmse:20.9772
[3]	validation_0-rmse:19.1871	validation_1-rmse:19.5182
[4]	validation_0-rmse:17.9094	validation_1-rmse:18.2466
[5]	validation_0-rmse:16.7982	validation_1-rmse:17.1434
[6]	validation_0-rmse:15.8372	validation_1-rmse:16.1914
[7]	validation_0-rmse:15.0163	validation_1-rmse:15.3786
[8]	validation_0-rmse:14.3185	validation_1-rmse:14.6897
[9]	validation_0-rmse:13.7192	validation_1-rmse:14.0985
[10]	validation_0-rmse:13.2092	validation_1-rmse:13.596
[11]	validation_0-rmse:12.7815	validation_1-rmse:13.1744
[12]	validation_0-rmse:12.4234	validation_1-rmse:12.8256
[13]	validation_0-rmse:12.1153	validation_1-rmse:12.533
[14]	validation_0-rmse:11.8656	validation_1-rmse:12.292
[15]	validation_0-rmse:11.6604	validation_1-rmse:12.0903
[16]	validation_0-rmse:11.4851	validation_1-rmse:11.9249
[17]	validation_0-rmse:11.3412	validation_1-rmse:11.7857
[18]	validation_0-rmse:11.2201	validation_1-rmse:11.6672
[19]	validation_0-rmse:11.1226	validation_1-rmse:11.5736
[20]	validation_0-rmse:11.0397	validation_1-rmse:11.4912
[21]	validation_0-rmse:10.9712	validation_1-rmse:11.4252
[22]	validation_0-rmse:10.912	validation_1-rmse:11.3704
[23]	validation_0-rmse:10.8579	validation_1-rmse:11.3227
[24]	validation_0-rmse:10.8115	validation_1-rmse:11.2836
[25]	validation_0-rmse:10.7723	validation_1-rmse:11.2482
[26]	validation_0-rmse:10.7426	validation_1-rmse:11.2214
[27]	validation_0-rmse:10.7105	validation_1-rmse:11.1987
[28]	validation_0-rmse:10.6878	validation_1-rmse:11.1796
[29]	validation_0-rmse:10.6715	validation_1-rmse:11.1644
[30]	validation_0-rmse:10.6507	validation_1-rmse:11.1482
[31]	validation_0-rmse:10.6315	validation_1-rmse:11.1434
[32]	validation_0-rmse:10.6155	validation_1-rmse:11.1269
[33]	validation_0-rmse:10.6019	validation_1-rmse:11.1178
[34]	validation_0-rmse:10.5878	validation_1-rmse:11.1077
[35]	validation_0-rmse:10.5695	validation_1-rmse:11.1027
[36]	validation_0-rmse:10.563	validation_1-rmse:11.0975
[37]	validation_0-rmse:10.5566	validation_1-rmse:11.0919
[38]	validation_0-rmse:10.5495	validation_1-rmse:11.0914
[39]	validation_0-rmse:10.5413	validation_1-rmse:11.0857
[40]	validation_0-rmse:10.5314	validation_1-rmse:11.0786
[41]	validation_0-rmse:10.5247	validation_1-rmse:11.0749
[42]	validation_0-rmse:10.5196	validation_1-rmse:11.072
[43]	validation_0-rmse:10.5155	validation_1-rmse:11.0692
[44]	validation_0-rmse:10.508	validation_1-rmse:11.064
[45]	validation_0-rmse:10.5025	validation_1-rmse:11.0639
[46]	validation_0-rmse:10.4903	validation_1-rmse:11.0635
[47]	validation_0-rmse:10.4869	validation_1-rmse:11.0609
[48]	validation_0-rmse:10.4815	validation_1-rmse:11.0629
[49]	validation_0-rmse:10.4727	validation_1-rmse:11.0585
[50]	validation_0-rmse:10.4694	validation_1-rmse:11.0587
[51]	validation_0-rmse:10.4582	validation_1-rmse:11.0685
[52]	validation_0-rmse:10.4536	validation_1-rmse:11.0652
[53]	validation_0-rmse:10.4509	validation_1-rmse:11.064
[54]	validation_0-rmse:10.4391	validation_1-rmse:11.0673
[55]	validation_0-rmse:10.4318	validation_1-rmse:11.0711
[56]	validation_0-rmse:10.4309	validation_1-rmse:11.0701
[57]	validation_0-rmse:10.4277	validation_1-rmse:11.0683
[58]	validation_0-rmse:10.4236	validation_1-rmse:11.0661
[59]	validation_0-rmse:10.4192	validation_1-rmse:11.0634
[60]	validation_0-rmse:10.4111	validation_1-rmse:11.0609
[61]	validation_0-rmse:10.4078	validation_1-rmse:11.0595
[62]	validation_0-rmse:10.4052	validation_1-rmse:11.0571
[63]	validation_0-rmse:10.4019	validation_1-rmse:11.0552
[64]	validation_0-rmse:10.3958	validation_1-rmse:11.0321
[65]	validation_0-rmse:10.3925	validation_1-rmse:11.0302
[66]	validation_0-rmse:10.3884	validation_1-rmse:11.0349
[67]	validation_0-rmse:10.3828	validation_1-rmse:11.0324
[68]	validation_0-rmse:10.3749	validation_1-rmse:11.0372
[69]	validation_0-rmse:10.3664	validation_1-rmse:11.0338
[70]	validation_0-rmse:10.3587	validation_1-rmse:11.0376
[71]	validation_0-rmse:10.3559	validation_1-rmse:11.0432
[72]	validation_0-rmse:10.349	validation_1-rmse:11.0451
[73]	validation_0-rmse:10.3441	validation_1-rmse:11.0437
[74]	validation_0-rmse:10.3405	validation_1-rmse:11.0451
[75]	validation_0-rmse:10.3377	validation_1-rmse:11.044
[76]	validation_0-rmse:10.3301	validation_1-rmse:11.056
[77]	validation_0-rmse:10.3265	validation_1-rmse:11.0579
[78]	validation_0-rmse:10.3241	validation_1-rmse:11.0565
[79]	validation_0-rmse:10.3219	validation_1-rmse:11.0553
[80]	validation_0-rmse:10.3207	validation_1-rmse:11.0546
[81]	validation_0-rmse:10.3144	validation_1-rmse:11.0683
[82]	validation_0-rmse:10.3116	validation_1-rmse:11.069
[83]	validation_0-rmse:10.3087	validation_1-rmse:11.0717
[84]	validation_0-rmse:10.3065	validation_1-rmse:11.0688
[85]	validation_0-rmse:10.3008	validation_1-rmse:11.081
[86]	validation_0-rmse:10.2977	validation_1-rmse:11.0796
[87]	validation_0-rmse:10.2955	validation_1-rmse:11.0793
[88]	validation_0-rmse:10.2927	validation_1-rmse:11.0847
[89]	validation_0-rmse:10.2911	validation_1-rmse:11.0797
[90]	validation_0-rmse:10.2901	validation_1-rmse:11.0789
[91]	validation_0-rmse:10.2855	validation_1-rmse:11.0742
[92]	validation_0-rmse:10.2841	validation_1-rmse:11.0742
[93]	validation_0-rmse:10.2821	validation_1-rmse:11.0754
[94]	validation_0-rmse:10.2811	validation_1-rmse:11.0741
[95]	validation_0-rmse:10.2787	validation_1-rmse:11.0729
[96]	validation_0-rmse:10.2756	validation_1-rmse:11.0705
[97]	validation_0-rmse:10.274	validation_1-rmse:11.0694
[98]	validation_0-rmse:10.2697	validation_1-rmse:11.0649
[99]	validation_0-rmse:10.2684	validation_1-rmse:11.0646
[100]	validation_0-rmse:10.2666	validation_1-rmse:11.0656
[101]	validation_0-rmse:10.2595	validation_1-rmse:11.0821
[102]	validation_0-rmse:10.2565	validation_1-rmse:11.0855
[103]	validation_0-rmse:10.2542	validation_1-rmse:11.0836
[104]	validation_0-rmse:10.2487	validation_1-rmse:11.09
[105]	validation_0-rmse:10.2476	validation_1-rmse:11.0898
[106]	validation_0-rmse:10.2464	validation_1-rmse:11.0988
[107]	validation_0-rmse:10.2433	validation_1-rmse:11.0977
[108]	validation_0-rmse:10.243	validation_1-rmse:11.0982
[109]	validation_0-rmse:10.2413	validation_1-rmse:11.0925
[110]	validation_0-rmse:10.2381	validation_1-rmse:11.092
[111]	validation_0-rmse:10.2373	validation_1-rmse:11.0898
[112]	validation_0-rmse:10.2281	validation_1-rmse:11.0897
[113]	validation_0-rmse:10.2241	validation_1-rmse:11.0954
[114]	validation_0-rmse:10.219	validation_1-rmse:11.094
[115]	validation_0-rmse:10.2167	validation_1-rmse:11.0921
Stopping. Best iteration:
[65]	validation_0-rmse:10.3925	validation_1-rmse:11.0302

[0]	validation_0-rmse:24.2313	validation_1-rmse:24.5479
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3163	validation_1-rmse:22.6379
[2]	validation_0-rmse:20.6318	validation_1-rmse:20.9626
[3]	validation_0-rmse:19.1548	validation_1-rmse:19.5006
[4]	validation_0-rmse:17.8661	validation_1-rmse:18.2276
[5]	validation_0-rmse:16.7434	validation_1-rmse:17.1202
[6]	validation_0-rmse:15.7738	validation_1-rmse:16.1639
[7]	validation_0-rmse:14.9361	validation_1-rmse:15.345
[8]	validation_0-rmse:14.2252	validation_1-rmse:14.6512
[9]	validation_0-rmse:13.6167	validation_1-rmse:14.0565
[10]	validation_0-rmse:13.0963	validation_1-rmse:13.5486
[11]	validation_0-rmse:12.6611	validation_1-rmse:13.127
[12]	validation_0-rmse:12.2937	validation_1-rmse:12.7745
[13]	validation_0-rmse:11.9776	validation_1-rmse:12.4767
[14]	validation_0-rmse:11.7185	validation_1-rmse:12.2308
[15]	validation_0-rmse:11.5058	validation_1-rmse:12.0279
[16]	validation_0-rmse:11.3189	validation_1-rmse:11.8558
[17]	validation_0-rmse:11.1698	validation_1-rmse:11.7145
[18]	validation_0-rmse:11.0441	validation_1-rmse:11.5963
[19]	validation_0-rmse:10.9435	validation_1-rmse:11.4988
[20]	validation_0-rmse:10.8482	validation_1-rmse:11.4178
[21]	validation_0-rmse:10.7681	validation_1-rmse:11.3523
[22]	validation_0-rmse:10.7049	validation_1-rmse:11.2961
[23]	validation_0-rmse:10.6516	validation_1-rmse:11.2522
[24]	validation_0-rmse:10.6073	validation_1-rmse:11.2117
[25]	validation_0-rmse:10.565	validation_1-rmse:11.1765
[26]	validation_0-rmse:10.5253	validation_1-rmse:11.1474
[27]	validation_0-rmse:10.4941	validation_1-rmse:11.1328
[28]	validation_0-rmse:10.4667	validation_1-rmse:11.1129
[29]	validation_0-rmse:10.4418	validation_1-rmse:11.0912
[30]	validation_0-rmse:10.4183	validation_1-rmse:11.0793
[31]	validation_0-rmse:10.3994	validation_1-rmse:11.0717
[32]	validation_0-rmse:10.3825	validation_1-rmse:11.0638
[33]	validation_0-rmse:10.3672	validation_1-rmse:11.054
[34]	validation_0-rmse:10.3491	validation_1-rmse:11.0429
[35]	validation_0-rmse:10.3354	validation_1-rmse:11.0361
[36]	validation_0-rmse:10.3224	validation_1-rmse:11.0278
[37]	validation_0-rmse:10.3152	validation_1-rmse:11.0212
[38]	validation_0-rmse:10.3057	validation_1-rmse:11.0192
[39]	validation_0-rmse:10.2963	validation_1-rmse:11.0147
[40]	validation_0-rmse:10.2831	validation_1-rmse:11.0134
[41]	validation_0-rmse:10.2755	validation_1-rmse:11.012
[42]	validation_0-rmse:10.2694	validation_1-rmse:11.0077
[43]	validation_0-rmse:10.263	validation_1-rmse:11.0049
[44]	validation_0-rmse:10.2535	validation_1-rmse:10.9999
[45]	validation_0-rmse:10.2442	validation_1-rmse:11.0011
[46]	validation_0-rmse:10.23	validation_1-rmse:10.9989
[47]	validation_0-rmse:10.2244	validation_1-rmse:10.9955
[48]	validation_0-rmse:10.2188	validation_1-rmse:10.9947
[49]	validation_0-rmse:10.2117	validation_1-rmse:10.9926
[50]	validation_0-rmse:10.2065	validation_1-rmse:10.9967
[51]	validation_0-rmse:10.1956	validation_1-rmse:11.0046
[52]	validation_0-rmse:10.1907	validation_1-rmse:11.0029
[53]	validation_0-rmse:10.1887	validation_1-rmse:11.0018
[54]	validation_0-rmse:10.1779	validation_1-rmse:11.007
[55]	validation_0-rmse:10.1723	validation_1-rmse:11.0117
[56]	validation_0-rmse:10.1711	validation_1-rmse:11.0166
[57]	validation_0-rmse:10.1592	validation_1-rmse:11.0164
[58]	validation_0-rmse:10.1529	validation_1-rmse:11.0134
[59]	validation_0-rmse:10.1399	validation_1-rmse:11.0059
[60]	validation_0-rmse:10.1393	validation_1-rmse:11.0066
[61]	validation_0-rmse:10.1347	validation_1-rmse:11.0078
[62]	validation_0-rmse:10.1269	validation_1-rmse:11.0031
[63]	validation_0-rmse:10.1194	validation_1-rmse:11.0023
[64]	validation_0-rmse:10.1149	validation_1-rmse:11.0014
[65]	validation_0-rmse:10.1117	validation_1-rmse:11.002
[66]	validation_0-rmse:10.1087	validation_1-rmse:11.0016
[67]	validation_0-rmse:10.1043	validation_1-rmse:11.0008
[68]	validation_0-rmse:10.1001	validation_1-rmse:10.9955
[69]	validation_0-rmse:10.091	validation_1-rmse:10.9976
[70]	validation_0-rmse:10.0853	validation_1-rmse:10.9967
[71]	validation_0-rmse:10.081	validation_1-rmse:11.0012
[72]	validation_0-rmse:10.07	validation_1-rmse:11.0051
[73]	validation_0-rmse:10.0668	validation_1-rmse:11.0025
[74]	validation_0-rmse:10.0622	validation_1-rmse:11.0017
[75]	validation_0-rmse:10.0572	validation_1-rmse:10.9999
[76]	validation_0-rmse:10.0528	validation_1-rmse:11.0019
[77]	validation_0-rmse:10.0485	validation_1-rmse:11.0087
[78]	validation_0-rmse:10.0434	validation_1-rmse:11.0098
[79]	validation_0-rmse:10.0431	validation_1-rmse:11.0095
[80]	validation_0-rmse:10.035	validation_1-rmse:11.0055
[81]	validation_0-rmse:10.0323	validation_1-rmse:11.0078
[82]	validation_0-rmse:10.0272	validation_1-rmse:11.0053
[83]	validation_0-rmse:10.024	validation_1-rmse:11.0049
[84]	validation_0-rmse:10.0212	validation_1-rmse:11.0036
[85]	validation_0-rmse:10.0159	validation_1-rmse:11.0062
[86]	validation_0-rmse:10.0102	validation_1-rmse:11.0025
[87]	validation_0-rmse:10.0063	validation_1-rmse:11.0015
[88]	validation_0-rmse:10.0046	validation_1-rmse:10.9996
[89]	validation_0-rmse:10.0018	validation_1-rmse:10.9944
[90]	validation_0-rmse:9.99709	validation_1-rmse:10.9921
[91]	validation_0-rmse:9.98844	validation_1-rmse:10.9902
[92]	validation_0-rmse:9.98498	validation_1-rmse:10.9886
[93]	validation_0-rmse:9.98294	validation_1-rmse:10.9871
[94]	validation_0-rmse:9.97683	validation_1-rmse:10.9843
[95]	validation_0-rmse:9.97259	validation_1-rmse:10.9837
[96]	validation_0-rmse:9.97161	validation_1-rmse:10.9829
[97]	validation_0-rmse:9.96985	validation_1-rmse:10.9848
[98]	validation_0-rmse:9.96638	validation_1-rmse:10.9868
[99]	validation_0-rmse:9.95866	validation_1-rmse:10.983
[100]	validation_0-rmse:9.95723	validation_1-rmse:10.982
[101]	validation_0-rmse:9.95335	validation_1-rmse:10.9929
[102]	validation_0-rmse:9.95037	validation_1-rmse:11.0035
[103]	validation_0-rmse:9.94645	validation_1-rmse:11.0013
[104]	validation_0-rmse:9.94284	validation_1-rmse:11.0005
[105]	validation_0-rmse:9.93918	validation_1-rmse:10.9995
[106]	validation_0-rmse:9.93782	validation_1-rmse:11.0098
[107]	validation_0-rmse:9.93352	validation_1-rmse:11.0122
[108]	validation_0-rmse:9.93091	validation_1-rmse:11.0095
[109]	validation_0-rmse:9.92907	validation_1-rmse:11.0054
[110]	validation_0-rmse:9.92582	validation_1-rmse:11.0046
[111]	validation_0-rmse:9.92511	validation_1-rmse:11.0046
[112]	validation_0-rmse:9.9169	validation_1-rmse:11.0076
[113]	validation_0-rmse:9.91262	validation_1-rmse:11.0043
[114]	validation_0-rmse:9.90712	validation_1-rmse:10.9959
^C---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
<ipython-input-105-5fe786759b95> in <module>()
     78         X_test=X_test,
     79         y_train=y_train,
---> 80         y_test=y_test
     81     )

<ipython-input-105-5fe786759b95> in regression_params_opt(origin_model_d, current_model, X_train, X_test, y_train, y_test)
     46                 algo=tpe.suggest,
     47                 max_evals=max_eval,
---> 48                 trials=trials)
     49     print(best)
     50     print(best.keys())

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    305             verbose=verbose,
    306             catch_eval_exceptions=catch_eval_exceptions,
--> 307             return_argmin=return_argmin,
    308         )
    309

~/python3/lib/python3.6/site-packages/hyperopt/base.py in fmin(self, fn, space, algo, max_evals, rstate, verbose, pass_expr_memo_ctrl, catch_eval_exceptions, return_argmin)
    633             pass_expr_memo_ctrl=pass_expr_memo_ctrl,
    634             catch_eval_exceptions=catch_eval_exceptions,
--> 635             return_argmin=return_argmin)
    636
    637

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
    318                     verbose=verbose)
    319     rval.catch_eval_exceptions = catch_eval_exceptions
--> 320     rval.exhaust()
    321     if return_argmin:
    322         return trials.argmin

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in exhaust(self)
    197     def exhaust(self):
    198         n_done = len(self.trials)
--> 199         self.run(self.max_evals - n_done, block_until_done=self.async)
    200         self.trials.refresh()
    201         return self

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in run(self, N, block_until_done)
    171             else:
    172                 # -- loop over trials and do the jobs directly
--> 173                 self.serial_evaluate()
    174
    175             if stopped:

~/python3/lib/python3.6/site-packages/hyperopt/fmin.py in serial_evaluate(self, N)
     90                 ctrl = base.Ctrl(self.trials, current_trial=trial)
     91                 try:
---> 92                     result = self.domain.evaluate(spec, ctrl)
     93                 except Exception as e:
     94                     logger.info('job exception: %s' % str(e))

~/python3/lib/python3.6/site-packages/hyperopt/base.py in evaluate(self, config, ctrl, attach_attachments)
    838                 memo=memo,
    839                 print_node_on_error=self.rec_eval_print_node_on_error)
--> 840             rval = self.fn(pyll_rval)
    841
    842         if isinstance(rval, (float, int, np.number)):

<ipython-input-105-5fe786759b95> in rmse_score(params)
     35     def rmse_score(params):
     36         model_fit = origin_model_d[current_model][0](
---> 37             **params).fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test),],early_stopping_rounds=50)
     38         y_pred_train = model_fit.predict(X_test)
     39         loss = mean_squared_error(y_test, y_pred_train)**0.5

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/sklearn.py in fit(self, X, y, sample_weight, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model)
    291                               early_stopping_rounds=early_stopping_rounds,
    292                               evals_result=evals_result, obj=obj, feval=feval,
--> 293                               verbose_eval=verbose, xgb_model=xgb_model)
    294
    295         if evals_result:

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, learning_rates)
    202                            evals=evals,
    203                            obj=obj, feval=feval,
--> 204                            xgb_model=xgb_model, callbacks=callbacks)
    205
    206

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/training.py in _train_internal(params, dtrain, num_boost_round, evals, obj, feval, xgb_model, callbacks)
     72         # Skip the first update if it is a recovery step.
     73         if version % 2 == 0:
---> 74             bst.update(dtrain, i, obj)
     75             bst.save_rabit_checkpoint()
     76             version += 1

~/python3/lib/python3.6/site-packages/xgboost-0.7-py3.6.egg/xgboost/core.py in update(self, dtrain, iteration, fobj)
    896         if fobj is None:
    897             _check_call(_LIB.XGBoosterUpdateOneIter(self.handle, ctypes.c_int(iteration),
--> 898                                                     dtrain.handle))
    899         else:
    900             pred = self.predict(dtrain)

KeyboardInterrupt:

In [106]: ps
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-106-1714145ac4bb> in <module>()
----> 1 ps

NameError: name 'ps' is not defined

In [107]: ps
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-107-1714145ac4bb> in <module>()
----> 1 ps

NameError: name 'ps' is not defined

In [108]: aaa
Out[108]:
[(11.032122534656253,
  [0.5,
   0.6000000000000001,
   0.1,
   5,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.7,
   2.3000000000000003,
   0.6000000000000001,
   129]),
 (10.867711346768681,
  [0.75,
   1.7000000000000002,
   0.1,
   8,
   2.0,
   1500,
   -1,
   'reg:linear',
   3.1,
   3.6,
   0.75,
   263]),
 (11.056675532819281,
  [0.8,
   4.800000000000001,
   0.1,
   6,
   4.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.6,
   0.65,
   71]),
 (10.99684259948922,
  [0.4,
   1.0,
   0.1,
   5,
   4.0,
   1500,
   -1,
   'reg:linear',
   1.7000000000000002,
   3.6,
   0.9,
   131]),
 (10.789125157766302,
  [0.75,
   2.8000000000000003,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   0.7000000000000001,
   3.2,
   1.0,
   245]),
 (10.869080687503255,
  [0.55,
   1.6,
   0.1,
   7,
   2.0,
   1500,
   -1,
   'reg:linear',
   2.8000000000000003,
   1.9000000000000001,
   0.8,
   244]),
 (11.003548402352093,
  [0.55, 2.1, 0.1, 6, 2.0, 1500, -1, 'reg:linear', 0.0, 0.8, 0.65, 129]),
 (10.890873408731197,
  [0.7000000000000001,
   0.6000000000000001,
   0.1,
   9,
   3.0,
   1500,
   -1,
   'reg:linear',
   1.0,
   1.8,
   0.65,
   151]),
 (11.015703670319155,
  [0.75,
   2.9000000000000004,
   0.1,
   5,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.4000000000000004,
   2.2,
   0.9,
   104]),
 (10.840786277522257,
  [0.45,
   4.0,
   0.1,
   8,
   3.0,
   1500,
   -1,
   'reg:linear',
   3.0,
   2.4000000000000004,
   0.8,
   176])]

In [109]: train[col]
Out[109]:
        dow  year  month  day_of_week  holiday_flg  min_visitors  \
0         2  2016      1            6            0           7.0
1         3  2016      1            4            0           2.0
2         4  2016      1            0            0           4.0
3         5  2016      1            2            0           6.0
4         0  2016      1            1            0           2.0
5         1  2016      1            5            0           5.0
6         2  2016      1            6            0           7.0
7         3  2016      1            4            0           2.0
8         4  2016      1            0            0           4.0
9         5  2016      1            2            0           6.0
10        0  2016      1            1            0           2.0
11        1  2016      1            5            0           5.0
12        2  2016      1            6            0           7.0
13        3  2016      1            4            0           2.0
14        4  2016      1            0            0           4.0
15        5  2016      1            2            0           6.0
16        2  2016      2            6            0           7.0
17        3  2016      2            4            0           2.0
18        4  2016      2            0            0           4.0
19        5  2016      2            2            0           6.0
20        0  2016      2            1            0           2.0
21        1  2016      2            5            0           5.0
22        2  2016      2            6            0           7.0
23        3  2016      2            4            1           2.0
24        4  2016      2            0            0           4.0
25        5  2016      2            2            0           6.0
26        0  2016      2            1            0           2.0
27        1  2016      2            5            0           5.0
28        2  2016      2            6            0           7.0
29        3  2016      2            4            0           2.0
...     ...   ...    ...          ...          ...           ...
252078    4  2017      3            0            0           2.0
252079    5  2017      3            2            0           2.0
252080    1  2017      3            5            0           2.0
252081    2  2017      3            6            0           2.0
252082    3  2017      3            4            0           2.0
252083    4  2017      3            0            0           2.0
252084    5  2017      3            2            0           2.0
252085    0  2017      3            1            0           2.0
252086    1  2017      3            5            0           2.0
252087    2  2017      3            6            0           2.0
252088    3  2017      3            4            0           2.0
252089    4  2017      3            0            0           2.0
252090    5  2017      4            2            0           2.0
252091    0  2017      4            1            0           2.0
252092    1  2017      4            5            0           2.0
252093    2  2017      4            6            0           2.0
252094    3  2017      4            4            0           2.0
252095    4  2017      4            0            0           2.0
252096    5  2017      4            2            0           2.0
252097    0  2017      4            1            0           2.0
252098    1  2017      4            5            0           2.0
252099    2  2017      4            6            0           2.0
252100    3  2017      4            4            0           2.0
252101    4  2017      4            0            0           2.0
252102    5  2017      4            2            0           2.0
252103    1  2017      4            5            0           2.0
252104    2  2017      4            6            0           2.0
252105    3  2017      4            4            0           2.0
252106    4  2017      4            0            0           2.0
252107    5  2017      4            2            0           2.0

        mean_visitors  median_visitors  max_visitors  count_observations  \
0           23.843750             25.0          57.0                64.0
1           20.292308             21.0          54.0                65.0
2           34.738462             35.0          61.0                65.0
3           27.651515             27.0          53.0                66.0
4           13.754386             12.0          34.0                57.0
5           18.580645             19.0          35.0                62.0
6           23.843750             25.0          57.0                64.0
7           20.292308             21.0          54.0                65.0
8           34.738462             35.0          61.0                65.0
9           27.651515             27.0          53.0                66.0
10          13.754386             12.0          34.0                57.0
11          18.580645             19.0          35.0                62.0
12          23.843750             25.0          57.0                64.0
13          20.292308             21.0          54.0                65.0
14          34.738462             35.0          61.0                65.0
15          27.651515             27.0          53.0                66.0
16          23.843750             25.0          57.0                64.0
17          20.292308             21.0          54.0                65.0
18          34.738462             35.0          61.0                65.0
19          27.651515             27.0          53.0                66.0
20          13.754386             12.0          34.0                57.0
21          18.580645             19.0          35.0                62.0
22          23.843750             25.0          57.0                64.0
23          20.292308             21.0          54.0                65.0
24          34.738462             35.0          61.0                65.0
25          27.651515             27.0          53.0                66.0
26          13.754386             12.0          34.0                57.0
27          18.580645             19.0          35.0                62.0
28          23.843750             25.0          57.0                64.0
29          20.292308             21.0          54.0                65.0
...               ...              ...           ...                 ...
252078       5.738095              6.0           9.0                42.0
252079       5.289474              6.0           8.0                38.0
252080       5.615385              6.0          11.0                39.0
252081       6.575000              6.0          25.0                40.0
252082       5.394737              6.0           8.0                38.0
252083       5.738095              6.0           9.0                42.0
252084       5.289474              6.0           8.0                38.0
252085       4.794118              4.5           8.0                34.0
252086       5.615385              6.0          11.0                39.0
252087       6.575000              6.0          25.0                40.0
252088       5.394737              6.0           8.0                38.0
252089       5.738095              6.0           9.0                42.0
252090       5.289474              6.0           8.0                38.0
252091       4.794118              4.5           8.0                34.0
252092       5.615385              6.0          11.0                39.0
252093       6.575000              6.0          25.0                40.0
252094       5.394737              6.0           8.0                38.0
252095       5.738095              6.0           9.0                42.0
252096       5.289474              6.0           8.0                38.0
252097       4.794118              4.5           8.0                34.0
252098       5.615385              6.0          11.0                39.0
252099       6.575000              6.0          25.0                40.0
252100       5.394737              6.0           8.0                38.0
252101       5.738095              6.0           9.0                42.0
252102       5.289474              6.0           8.0                38.0
252103       5.615385              6.0          11.0                39.0
252104       6.575000              6.0          25.0                40.0
252105       5.394737              6.0           8.0                38.0
252106       5.738095              6.0           9.0                42.0
252107       5.289474              6.0           8.0                38.0

            ...        rs2_y  rv2_y  total_reserv_sum  total_reserv_mean  \
0           ...         -1.0   -1.0              -1.0               -1.0
1           ...         -1.0   -1.0              -1.0               -1.0
2           ...         -1.0   -1.0              -1.0               -1.0
3           ...         -1.0   -1.0              -1.0               -1.0
4           ...         -1.0   -1.0              -1.0               -1.0
5           ...         -1.0   -1.0              -1.0               -1.0
6           ...         -1.0   -1.0              -1.0               -1.0
7           ...         -1.0   -1.0              -1.0               -1.0
8           ...         -1.0   -1.0              -1.0               -1.0
9           ...         -1.0   -1.0              -1.0               -1.0
10          ...         -1.0   -1.0              -1.0               -1.0
11          ...         -1.0   -1.0              -1.0               -1.0
12          ...         -1.0   -1.0              -1.0               -1.0
13          ...         -1.0   -1.0              -1.0               -1.0
14          ...         -1.0   -1.0              -1.0               -1.0
15          ...         -1.0   -1.0              -1.0               -1.0
16          ...         -1.0   -1.0              -1.0               -1.0
17          ...         -1.0   -1.0              -1.0               -1.0
18          ...         -1.0   -1.0              -1.0               -1.0
19          ...         -1.0   -1.0              -1.0               -1.0
20          ...         -1.0   -1.0              -1.0               -1.0
21          ...         -1.0   -1.0              -1.0               -1.0
22          ...         -1.0   -1.0              -1.0               -1.0
23          ...         -1.0   -1.0              -1.0               -1.0
24          ...         -1.0   -1.0              -1.0               -1.0
25          ...         -1.0   -1.0              -1.0               -1.0
26          ...         -1.0   -1.0              -1.0               -1.0
27          ...         -1.0   -1.0              -1.0               -1.0
28          ...         -1.0   -1.0              -1.0               -1.0
29          ...         -1.0   -1.0              -1.0               -1.0
...         ...          ...    ...               ...                ...
252078      ...         -1.0   -1.0              -1.0               -1.0
252079      ...         -1.0   -1.0              -1.0               -1.0
252080      ...         -1.0   -1.0              -1.0               -1.0
252081      ...         -1.0   -1.0              -1.0               -1.0
252082      ...         -1.0   -1.0              -1.0               -1.0
252083      ...         -1.0   -1.0              -1.0               -1.0
252084      ...         -1.0   -1.0              -1.0               -1.0
252085      ...         -1.0   -1.0              -1.0               -1.0
252086      ...         -1.0   -1.0              -1.0               -1.0
252087      ...         -1.0   -1.0              -1.0               -1.0
252088      ...         -1.0   -1.0              -1.0               -1.0
252089      ...         -1.0   -1.0              -1.0               -1.0
252090      ...         -1.0   -1.0              -1.0               -1.0
252091      ...         -1.0   -1.0              -1.0               -1.0
252092      ...         -1.0   -1.0              -1.0               -1.0
252093      ...         -1.0   -1.0              -1.0               -1.0
252094      ...         -1.0   -1.0              -1.0               -1.0
252095      ...         -1.0   -1.0              -1.0               -1.0
252096      ...         -1.0   -1.0              -1.0               -1.0
252097      ...         -1.0   -1.0              -1.0               -1.0
252098      ...         -1.0   -1.0              -1.0               -1.0
252099      ...         -1.0   -1.0              -1.0               -1.0
252100      ...         -1.0   -1.0              -1.0               -1.0
252101      ...         -1.0   -1.0              -1.0               -1.0
252102      ...         -1.0   -1.0              -1.0               -1.0
252103      ...         -1.0   -1.0              -1.0               -1.0
252104      ...         -1.0   -1.0              -1.0               -1.0
252105      ...         -1.0   -1.0              -1.0               -1.0
252106      ...         -1.0   -1.0              -1.0               -1.0
252107      ...         -1.0   -1.0              -1.0               -1.0

        total_reserv_dt_diff_mean  date_int  var_max_lat  var_max_long  \
0                            -1.0  20160113     8.362564      4.521799
1                            -1.0  20160114     8.362564      4.521799
2                            -1.0  20160115     8.362564      4.521799
3                            -1.0  20160116     8.362564      4.521799
4                            -1.0  20160118     8.362564      4.521799
5                            -1.0  20160119     8.362564      4.521799
6                            -1.0  20160120     8.362564      4.521799
7                            -1.0  20160121     8.362564      4.521799
8                            -1.0  20160122     8.362564      4.521799
9                            -1.0  20160123     8.362564      4.521799
10                           -1.0  20160125     8.362564      4.521799
11                           -1.0  20160126     8.362564      4.521799
12                           -1.0  20160127     8.362564      4.521799
13                           -1.0  20160128     8.362564      4.521799
14                           -1.0  20160129     8.362564      4.521799
15                           -1.0  20160130     8.362564      4.521799
16                           -1.0  20160203     8.362564      4.521799
17                           -1.0  20160204     8.362564      4.521799
18                           -1.0  20160205     8.362564      4.521799
19                           -1.0  20160206     8.362564      4.521799
20                           -1.0  20160208     8.362564      4.521799
21                           -1.0  20160209     8.362564      4.521799
22                           -1.0  20160210     8.362564      4.521799
23                           -1.0  20160211     8.362564      4.521799
24                           -1.0  20160212     8.362564      4.521799
25                           -1.0  20160213     8.362564      4.521799
26                           -1.0  20160215     8.362564      4.521799
27                           -1.0  20160216     8.362564      4.521799
28                           -1.0  20160217     8.362564      4.521799
29                           -1.0  20160218     8.362564      4.521799
...                           ...       ...          ...           ...
252078                       -1.0  20170317     8.367414      4.562362
252079                       -1.0  20170318     8.367414      4.562362
252080                       -1.0  20170321     8.367414      4.562362
252081                       -1.0  20170322     8.367414      4.562362
252082                       -1.0  20170323     8.367414      4.562362
252083                       -1.0  20170324     8.367414      4.562362
252084                       -1.0  20170325     8.367414      4.562362
252085                       -1.0  20170327     8.367414      4.562362
252086                       -1.0  20170328     8.367414      4.562362
252087                       -1.0  20170329     8.367414      4.562362
252088                       -1.0  20170330     8.367414      4.562362
252089                       -1.0  20170331     8.367414      4.562362
252090                       -1.0  20170401     8.367414      4.562362
252091                       -1.0  20170403     8.367414      4.562362
252092                       -1.0  20170404     8.367414      4.562362
252093                       -1.0  20170405     8.367414      4.562362
252094                       -1.0  20170406     8.367414      4.562362
252095                       -1.0  20170407     8.367414      4.562362
252096                       -1.0  20170408     8.367414      4.562362
252097                       -1.0  20170410     8.367414      4.562362
252098                       -1.0  20170411     8.367414      4.562362
252099                       -1.0  20170412     8.367414      4.562362
252100                       -1.0  20170413     8.367414      4.562362
252101                       -1.0  20170414     8.367414      4.562362
252102                       -1.0  20170415     8.367414      4.562362
252103                       -1.0  20170418     8.367414      4.562362
252104                       -1.0  20170419     8.367414      4.562362
252105                       -1.0  20170420     8.367414      4.562362
252106                       -1.0  20170421     8.367414      4.562362
252107                       -1.0  20170422     8.367414      4.562362

        lon_plus_lat  air_store_id2
0         175.409667            603
1         175.409667            603
2         175.409667            603
3         175.409667            603
4         175.409667            603
5         175.409667            603
6         175.409667            603
7         175.409667            603
8         175.409667            603
9         175.409667            603
10        175.409667            603
11        175.409667            603
12        175.409667            603
13        175.409667            603
14        175.409667            603
15        175.409667            603
16        175.409667            603
17        175.409667            603
18        175.409667            603
19        175.409667            603
20        175.409667            603
21        175.409667            603
22        175.409667            603
23        175.409667            603
24        175.409667            603
25        175.409667            603
26        175.409667            603
27        175.409667            603
28        175.409667            603
29        175.409667            603
...              ...            ...
252078    175.364254             98
252079    175.364254             98
252080    175.364254             98
252081    175.364254             98
252082    175.364254             98
252083    175.364254             98
252084    175.364254             98
252085    175.364254             98
252086    175.364254             98
252087    175.364254             98
252088    175.364254             98
252089    175.364254             98
252090    175.364254             98
252091    175.364254             98
252092    175.364254             98
252093    175.364254             98
252094    175.364254             98
252095    175.364254             98
252096    175.364254             98
252097    175.364254             98
252098    175.364254             98
252099    175.364254             98
252100    175.364254             98
252101    175.364254             98
252102    175.364254             98
252103    175.364254             98
252104    175.364254             98
252105    175.364254             98
252106    175.364254             98
252107    175.364254             98

[252108 rows x 50 columns]

In [110]: y.shape
Out[110]: (252108,)

In [111]: y.shape
Out[111]: (252108,)

In [112]: %paste
def get_final_parameters(best_params, origin_model_d, current_model):
    """
    Fix the string parameters returned by hyperopt.
    Use the index given by hyperopt to find the real string value
    for a specific parameter
    """
    for element_p in list(best_params.keys()):
        if isinstance(
                origin_model_d[current_model][1][element_p],
                hyperopt.pyll.base.Apply):
            if origin_model_d[current_model][1][element_p].name == "float":
                pass
            elif origin_model_d[current_model][1][element_p].name == "switch":
                apply_obj = origin_model_d[current_model][1][element_p]
                literal_obj = apply_obj.pos_args[best_params[element_p] + 1]
                best_params[element_p] = literal_obj.obj
            else:
                pass
        else:
            pass
    return best_params

def regression_params_opt(
        origin_model_d,
        current_model,
        X_train,
        X_test,
        y_train,
        y_test):
    print(current_model)
    best = 0
    max_eval = 10
    trials = Trials()

    def rmse_score(params):
        model_fit = origin_model_d[current_model][0](
            **params).fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(X_test)
        loss = mean_squared_error(y_test, y_pred_train)**0.5
        params['best_iteration'] = model_fit.best_iteration
        list_result_hyperopt.append((loss, params))
        return {'loss': loss, 'status': STATUS_OK}

    best = fmin(rmse_score,
                origin_model_d[current_model][1],
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials)
    print(best)
    print(best.keys())
    print("\n")
    return get_final_parameters(best, origin_model_d, current_model)

if __name__ == '__main__':

    xgbr_d = {'gamma': hp.quniform('gamma', 0.0, 5.0, 0.1),
              'learning_rate': hp.choice('learning_rate', [0.1]),
              'colsample_bytree': hp.quniform('colsample_bytree',
                                              0.3,
                                              1.,
                                              0.05),
              'max_depth': hp.choice('max_depth', list(range(5, 10))),
              'min_child_weight': hp.quniform('min_child_weight', 1., 5., 1),
              'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
              'nthread': hp.choice('nthread', [-1]),
              'n_estimators': hp.choice('n_estimators', [1500]),
              'objective': hp.choice('objective', ['reg:linear']),
              'reg_lambda': hp.quniform('reg_alpha', 0.0, 4.0, 0.1),
              'reg_alpha': hp.quniform('reg_lambda', 0.0, 4.0, 0.1)}

    base_model = {"XGBRegressor": [XGBRegressor, xgbr_d]}

    list_result_hyperopt = []
    hyper_parametres = regression_params_opt(
        origin_model_d=base_model,
        current_model="XGBRegressor",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

loss,params = sorted(list_result_hyperopt)[0]

logging.info('5.1 XGBRegressor with hyperopt nicolas parameters')
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

best_iteration = params['best_iteration']
del params['best_iteration']

params['n_estimators'] = best_iteration*5
params['learning_rate'] = 0.02
xgb_hyperopt1 = XGBRegressor(**params)
xgb_hyperopt1.fit(X_train, y_train)
xgbpred1 = xgb_hyperopt1.predict(X_test)
train_error_xgb_hyperopt1 = round(mean_squared_error(y_train, xgb_hyperopt1.predict(X_train)), 3)
test_error_xgb_hyperopt1 = round(mean_squared_error(y_test, xgb_hyperopt1.predict(X_test)), 3)
print("train error: {}".format(train_error_xgb_hyperopt1))
print("test error: {}".format(test_error_xgb_hyperopt1))
print("Hyperopt error:".format(loss))

## -- End pasted text --
XGBRegressor
[0]	validation_0-rmse:24.4533	validation_1-rmse:24.7724
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.5097	validation_1-rmse:22.8364
[2]	validation_0-rmse:20.8045	validation_1-rmse:21.1425
[3]	validation_0-rmse:19.4321	validation_1-rmse:19.7808
[4]	validation_0-rmse:18.4326	validation_1-rmse:18.7964
[5]	validation_0-rmse:17.4098	validation_1-rmse:17.7826
[6]	validation_0-rmse:16.448	validation_1-rmse:16.8315
[7]	validation_0-rmse:15.6867	validation_1-rmse:16.0861
[8]	validation_0-rmse:14.8918	validation_1-rmse:15.3037
[9]	validation_0-rmse:14.2068	validation_1-rmse:14.632
[10]	validation_0-rmse:13.6188	validation_1-rmse:14.0549
[11]	validation_0-rmse:13.1236	validation_1-rmse:13.5672
[12]	validation_0-rmse:12.8065	validation_1-rmse:13.2576
[13]	validation_0-rmse:12.5746	validation_1-rmse:13.0328
[14]	validation_0-rmse:12.3032	validation_1-rmse:12.7688
[15]	validation_0-rmse:12.0262	validation_1-rmse:12.4986
[16]	validation_0-rmse:11.8382	validation_1-rmse:12.3186
[17]	validation_0-rmse:11.6337	validation_1-rmse:12.1203
[18]	validation_0-rmse:11.5016	validation_1-rmse:11.9911
[19]	validation_0-rmse:11.3531	validation_1-rmse:11.8475
[20]	validation_0-rmse:11.273	validation_1-rmse:11.7722
[21]	validation_0-rmse:11.1957	validation_1-rmse:11.6958
[22]	validation_0-rmse:11.1505	validation_1-rmse:11.6539
[23]	validation_0-rmse:11.0597	validation_1-rmse:11.5652
[24]	validation_0-rmse:11.0197	validation_1-rmse:11.5275
[25]	validation_0-rmse:10.9463	validation_1-rmse:11.4574
[26]	validation_0-rmse:10.9191	validation_1-rmse:11.4327
[27]	validation_0-rmse:10.8927	validation_1-rmse:11.4085
[28]	validation_0-rmse:10.8594	validation_1-rmse:11.3791
[29]	validation_0-rmse:10.8387	validation_1-rmse:11.3627
[30]	validation_0-rmse:10.8133	validation_1-rmse:11.3385
[31]	validation_0-rmse:10.7716	validation_1-rmse:11.2995
[32]	validation_0-rmse:10.7584	validation_1-rmse:11.2893
[33]	validation_0-rmse:10.7226	validation_1-rmse:11.2571
[34]	validation_0-rmse:10.6952	validation_1-rmse:11.2354
[35]	validation_0-rmse:10.6848	validation_1-rmse:11.2273
[36]	validation_0-rmse:10.664	validation_1-rmse:11.2078
[37]	validation_0-rmse:10.6445	validation_1-rmse:11.1891
[38]	validation_0-rmse:10.6	validation_1-rmse:11.1769
[39]	validation_0-rmse:10.585	validation_1-rmse:11.1639
[40]	validation_0-rmse:10.5668	validation_1-rmse:11.1521
[41]	validation_0-rmse:10.5634	validation_1-rmse:11.1516
[42]	validation_0-rmse:10.5544	validation_1-rmse:11.1448
[43]	validation_0-rmse:10.5423	validation_1-rmse:11.1366
[44]	validation_0-rmse:10.5279	validation_1-rmse:11.1339
[45]	validation_0-rmse:10.5177	validation_1-rmse:11.1302
[46]	validation_0-rmse:10.5071	validation_1-rmse:11.1228
[47]	validation_0-rmse:10.5025	validation_1-rmse:11.1189
[48]	validation_0-rmse:10.488	validation_1-rmse:11.1068
[49]	validation_0-rmse:10.4833	validation_1-rmse:11.1037
[50]	validation_0-rmse:10.4731	validation_1-rmse:11.1069
[51]	validation_0-rmse:10.4708	validation_1-rmse:11.1057
[52]	validation_0-rmse:10.459	validation_1-rmse:11.0966
[53]	validation_0-rmse:10.4455	validation_1-rmse:11.0859
[54]	validation_0-rmse:10.4337	validation_1-rmse:11.0877
[55]	validation_0-rmse:10.4256	validation_1-rmse:11.0856
[56]	validation_0-rmse:10.4172	validation_1-rmse:11.0784
[57]	validation_0-rmse:10.4115	validation_1-rmse:11.0782
[58]	validation_0-rmse:10.4062	validation_1-rmse:11.0757
[59]	validation_0-rmse:10.4042	validation_1-rmse:11.0742
[60]	validation_0-rmse:10.3946	validation_1-rmse:11.0711
[61]	validation_0-rmse:10.3811	validation_1-rmse:11.0644
[62]	validation_0-rmse:10.3784	validation_1-rmse:11.0632
[63]	validation_0-rmse:10.3712	validation_1-rmse:11.0602
[64]	validation_0-rmse:10.3687	validation_1-rmse:11.0597
[65]	validation_0-rmse:10.3655	validation_1-rmse:11.0584
[66]	validation_0-rmse:10.3611	validation_1-rmse:11.0575
[67]	validation_0-rmse:10.348	validation_1-rmse:11.0533
[68]	validation_0-rmse:10.3387	validation_1-rmse:11.0464
[69]	validation_0-rmse:10.3347	validation_1-rmse:11.0475
[70]	validation_0-rmse:10.3293	validation_1-rmse:11.043
[71]	validation_0-rmse:10.2968	validation_1-rmse:11.0563
[72]	validation_0-rmse:10.2939	validation_1-rmse:11.0541
[73]	validation_0-rmse:10.2912	validation_1-rmse:11.0549
[74]	validation_0-rmse:10.2843	validation_1-rmse:11.0497
[75]	validation_0-rmse:10.2737	validation_1-rmse:11.0531
[76]	validation_0-rmse:10.2672	validation_1-rmse:11.0484
[77]	validation_0-rmse:10.2641	validation_1-rmse:11.0471
[78]	validation_0-rmse:10.2572	validation_1-rmse:11.0424
[79]	validation_0-rmse:10.2538	validation_1-rmse:11.0408
[80]	validation_0-rmse:10.2507	validation_1-rmse:11.039
[81]	validation_0-rmse:10.2493	validation_1-rmse:11.0385
[82]	validation_0-rmse:10.2421	validation_1-rmse:11.0375
[83]	validation_0-rmse:10.2399	validation_1-rmse:11.0352
[84]	validation_0-rmse:10.2343	validation_1-rmse:11.0324
[85]	validation_0-rmse:10.2292	validation_1-rmse:11.0307
[86]	validation_0-rmse:10.2266	validation_1-rmse:11.0296
[87]	validation_0-rmse:10.222	validation_1-rmse:11.026
[88]	validation_0-rmse:10.2182	validation_1-rmse:11.0266
[89]	validation_0-rmse:10.2114	validation_1-rmse:11.0241
[90]	validation_0-rmse:10.2087	validation_1-rmse:11.0224
[91]	validation_0-rmse:10.2058	validation_1-rmse:11.0223
[92]	validation_0-rmse:10.2038	validation_1-rmse:11.0216
[93]	validation_0-rmse:10.1975	validation_1-rmse:11.0275
[94]	validation_0-rmse:10.1943	validation_1-rmse:11.0286
[95]	validation_0-rmse:10.1924	validation_1-rmse:11.0268
[96]	validation_0-rmse:10.1906	validation_1-rmse:11.026
[97]	validation_0-rmse:10.1864	validation_1-rmse:11.0236
[98]	validation_0-rmse:10.1788	validation_1-rmse:11.0203
[99]	validation_0-rmse:10.1774	validation_1-rmse:11.0195
[100]	validation_0-rmse:10.1744	validation_1-rmse:11.0205
[101]	validation_0-rmse:10.172	validation_1-rmse:11.0197
[102]	validation_0-rmse:10.1683	validation_1-rmse:11.0174
[103]	validation_0-rmse:10.1667	validation_1-rmse:11.0166
[104]	validation_0-rmse:10.1519	validation_1-rmse:11.0128
[105]	validation_0-rmse:10.1506	validation_1-rmse:11.0123
[106]	validation_0-rmse:10.1473	validation_1-rmse:11.0114
[107]	validation_0-rmse:10.1441	validation_1-rmse:11.0102
[108]	validation_0-rmse:10.1436	validation_1-rmse:11.0106
[109]	validation_0-rmse:10.1419	validation_1-rmse:11.0145
[110]	validation_0-rmse:10.1372	validation_1-rmse:11.0107
[111]	validation_0-rmse:10.1351	validation_1-rmse:11.0098
[112]	validation_0-rmse:10.1325	validation_1-rmse:11.0081
[113]	validation_0-rmse:10.13	validation_1-rmse:11.0079
[114]	validation_0-rmse:10.1284	validation_1-rmse:11.0064
[115]	validation_0-rmse:10.1268	validation_1-rmse:11.0059
[116]	validation_0-rmse:10.1237	validation_1-rmse:11.0037
[117]	validation_0-rmse:10.1221	validation_1-rmse:11.0029
[118]	validation_0-rmse:10.1212	validation_1-rmse:11.0029
[119]	validation_0-rmse:10.1203	validation_1-rmse:11.0027
[120]	validation_0-rmse:10.1195	validation_1-rmse:11.002
[121]	validation_0-rmse:10.1183	validation_1-rmse:11.0018
[122]	validation_0-rmse:10.1122	validation_1-rmse:10.9988
[123]	validation_0-rmse:10.1104	validation_1-rmse:10.9988
[124]	validation_0-rmse:10.1089	validation_1-rmse:10.9989
[125]	validation_0-rmse:10.107	validation_1-rmse:10.9982
[126]	validation_0-rmse:10.1055	validation_1-rmse:10.9988
[127]	validation_0-rmse:10.0998	validation_1-rmse:10.997
[128]	validation_0-rmse:10.0975	validation_1-rmse:10.9959
[129]	validation_0-rmse:10.0806	validation_1-rmse:10.9921
[130]	validation_0-rmse:10.0793	validation_1-rmse:10.9921
[131]	validation_0-rmse:10.0706	validation_1-rmse:10.9899
[132]	validation_0-rmse:10.0694	validation_1-rmse:10.9921
[133]	validation_0-rmse:10.0683	validation_1-rmse:10.9932
[134]	validation_0-rmse:10.0672	validation_1-rmse:10.9949
[135]	validation_0-rmse:10.0635	validation_1-rmse:10.993
[136]	validation_0-rmse:10.0623	validation_1-rmse:10.9928
[137]	validation_0-rmse:10.0609	validation_1-rmse:10.9927
[138]	validation_0-rmse:10.0601	validation_1-rmse:10.9931
[139]	validation_0-rmse:10.0597	validation_1-rmse:10.9929
[140]	validation_0-rmse:10.0568	validation_1-rmse:10.9902
[141]	validation_0-rmse:10.0551	validation_1-rmse:10.9899
[142]	validation_0-rmse:10.03	validation_1-rmse:10.9876
[143]	validation_0-rmse:10.0283	validation_1-rmse:10.987
[144]	validation_0-rmse:10.0234	validation_1-rmse:10.9861
[145]	validation_0-rmse:10.0204	validation_1-rmse:10.9836
[146]	validation_0-rmse:10.02	validation_1-rmse:10.9842
[147]	validation_0-rmse:10.0192	validation_1-rmse:10.9844
[148]	validation_0-rmse:10.0166	validation_1-rmse:10.9849
[149]	validation_0-rmse:10.0149	validation_1-rmse:10.9846
[150]	validation_0-rmse:10.0124	validation_1-rmse:10.9843
[151]	validation_0-rmse:10.0123	validation_1-rmse:10.9829
[152]	validation_0-rmse:10.0109	validation_1-rmse:10.9821
[153]	validation_0-rmse:10.0098	validation_1-rmse:10.9815
[154]	validation_0-rmse:9.98904	validation_1-rmse:10.979
[155]	validation_0-rmse:9.98748	validation_1-rmse:10.978
[156]	validation_0-rmse:9.9861	validation_1-rmse:10.9773
[157]	validation_0-rmse:9.98499	validation_1-rmse:10.9773
[158]	validation_0-rmse:9.97953	validation_1-rmse:10.974
[159]	validation_0-rmse:9.9784	validation_1-rmse:10.9737
[160]	validation_0-rmse:9.9778	validation_1-rmse:10.9728
[161]	validation_0-rmse:9.97679	validation_1-rmse:10.9739
[162]	validation_0-rmse:9.96535	validation_1-rmse:10.978
[163]	validation_0-rmse:9.95649	validation_1-rmse:10.9789
[164]	validation_0-rmse:9.95175	validation_1-rmse:10.9825
[165]	validation_0-rmse:9.9502	validation_1-rmse:10.9816
[166]	validation_0-rmse:9.94914	validation_1-rmse:10.9813
[167]	validation_0-rmse:9.94777	validation_1-rmse:10.9806
[168]	validation_0-rmse:9.942	validation_1-rmse:10.9755
[169]	validation_0-rmse:9.94056	validation_1-rmse:10.9752
[170]	validation_0-rmse:9.93981	validation_1-rmse:10.9744
[171]	validation_0-rmse:9.92258	validation_1-rmse:10.9899
[172]	validation_0-rmse:9.92175	validation_1-rmse:10.9905
[173]	validation_0-rmse:9.90804	validation_1-rmse:10.9879
[174]	validation_0-rmse:9.90676	validation_1-rmse:10.9877
[175]	validation_0-rmse:9.90431	validation_1-rmse:10.9864
[176]	validation_0-rmse:9.89979	validation_1-rmse:10.9839
[177]	validation_0-rmse:9.89917	validation_1-rmse:10.9836
[178]	validation_0-rmse:9.8952	validation_1-rmse:10.9896
[179]	validation_0-rmse:9.89418	validation_1-rmse:10.9888
[180]	validation_0-rmse:9.89372	validation_1-rmse:10.9912
[181]	validation_0-rmse:9.89165	validation_1-rmse:10.9908
[182]	validation_0-rmse:9.89127	validation_1-rmse:10.991
[183]	validation_0-rmse:9.89038	validation_1-rmse:10.9912
[184]	validation_0-rmse:9.889	validation_1-rmse:10.9912
[185]	validation_0-rmse:9.88834	validation_1-rmse:10.991
[186]	validation_0-rmse:9.88764	validation_1-rmse:10.9909
[187]	validation_0-rmse:9.88541	validation_1-rmse:10.9894
[188]	validation_0-rmse:9.88353	validation_1-rmse:10.9884
[189]	validation_0-rmse:9.88322	validation_1-rmse:10.9874
[190]	validation_0-rmse:9.88292	validation_1-rmse:10.9874
[191]	validation_0-rmse:9.8808	validation_1-rmse:10.987
[192]	validation_0-rmse:9.87915	validation_1-rmse:10.9864
[193]	validation_0-rmse:9.86709	validation_1-rmse:10.9856
[194]	validation_0-rmse:9.86096	validation_1-rmse:10.984
[195]	validation_0-rmse:9.86075	validation_1-rmse:10.9839
[196]	validation_0-rmse:9.86052	validation_1-rmse:10.9844
[197]	validation_0-rmse:9.85938	validation_1-rmse:10.9841
[198]	validation_0-rmse:9.85724	validation_1-rmse:10.9832
[199]	validation_0-rmse:9.85695	validation_1-rmse:10.9813
[200]	validation_0-rmse:9.85166	validation_1-rmse:10.9812
[201]	validation_0-rmse:9.8501	validation_1-rmse:10.9808
[202]	validation_0-rmse:9.84434	validation_1-rmse:10.9808
[203]	validation_0-rmse:9.84398	validation_1-rmse:10.9815
[204]	validation_0-rmse:9.84116	validation_1-rmse:10.9793
[205]	validation_0-rmse:9.83825	validation_1-rmse:10.978
[206]	validation_0-rmse:9.83779	validation_1-rmse:10.9783
[207]	validation_0-rmse:9.83441	validation_1-rmse:10.9763
[208]	validation_0-rmse:9.83327	validation_1-rmse:10.9752
[209]	validation_0-rmse:9.83138	validation_1-rmse:10.9745
[210]	validation_0-rmse:9.82886	validation_1-rmse:10.9724
[211]	validation_0-rmse:9.8263	validation_1-rmse:10.9705
[212]	validation_0-rmse:9.82602	validation_1-rmse:10.9702
[213]	validation_0-rmse:9.8255	validation_1-rmse:10.9697
[214]	validation_0-rmse:9.82511	validation_1-rmse:10.9701
[215]	validation_0-rmse:9.82301	validation_1-rmse:10.9705
[216]	validation_0-rmse:9.82056	validation_1-rmse:10.9699
[217]	validation_0-rmse:9.81857	validation_1-rmse:10.9697
[218]	validation_0-rmse:9.81797	validation_1-rmse:10.9694
[219]	validation_0-rmse:9.81638	validation_1-rmse:10.9694
[220]	validation_0-rmse:9.81626	validation_1-rmse:10.9694
[221]	validation_0-rmse:9.81539	validation_1-rmse:10.9696
[222]	validation_0-rmse:9.81156	validation_1-rmse:10.9681
[223]	validation_0-rmse:9.81008	validation_1-rmse:10.9722
[224]	validation_0-rmse:9.81043	validation_1-rmse:10.9696
[225]	validation_0-rmse:9.81025	validation_1-rmse:10.9689
[226]	validation_0-rmse:9.81008	validation_1-rmse:10.9689
[227]	validation_0-rmse:9.80957	validation_1-rmse:10.9688
[228]	validation_0-rmse:9.8081	validation_1-rmse:10.9684
[229]	validation_0-rmse:9.80694	validation_1-rmse:10.9696
[230]	validation_0-rmse:9.79524	validation_1-rmse:10.9681
[231]	validation_0-rmse:9.79273	validation_1-rmse:10.9664
[232]	validation_0-rmse:9.78929	validation_1-rmse:10.9685
[233]	validation_0-rmse:9.78675	validation_1-rmse:10.9672
[234]	validation_0-rmse:9.77505	validation_1-rmse:10.9842
[235]	validation_0-rmse:9.77159	validation_1-rmse:10.9821
[236]	validation_0-rmse:9.76957	validation_1-rmse:10.9812
[237]	validation_0-rmse:9.76707	validation_1-rmse:10.9804
[238]	validation_0-rmse:9.76132	validation_1-rmse:10.9771
[239]	validation_0-rmse:9.76084	validation_1-rmse:10.9775
[240]	validation_0-rmse:9.76009	validation_1-rmse:10.977
[241]	validation_0-rmse:9.75922	validation_1-rmse:10.978
[242]	validation_0-rmse:9.75627	validation_1-rmse:10.9814
[243]	validation_0-rmse:9.75541	validation_1-rmse:10.9815
[244]	validation_0-rmse:9.75423	validation_1-rmse:10.981
[245]	validation_0-rmse:9.75228	validation_1-rmse:10.9805
[246]	validation_0-rmse:9.75135	validation_1-rmse:10.98
[247]	validation_0-rmse:9.74268	validation_1-rmse:10.9848
[248]	validation_0-rmse:9.74198	validation_1-rmse:10.9859
[249]	validation_0-rmse:9.74197	validation_1-rmse:10.987
[250]	validation_0-rmse:9.74184	validation_1-rmse:10.9871
[251]	validation_0-rmse:9.74053	validation_1-rmse:10.9872
[252]	validation_0-rmse:9.74033	validation_1-rmse:10.9875
[253]	validation_0-rmse:9.74018	validation_1-rmse:10.9872
[254]	validation_0-rmse:9.73956	validation_1-rmse:10.9884
[255]	validation_0-rmse:9.73609	validation_1-rmse:10.9872
[256]	validation_0-rmse:9.73517	validation_1-rmse:10.9864
[257]	validation_0-rmse:9.73426	validation_1-rmse:10.986
[258]	validation_0-rmse:9.73368	validation_1-rmse:10.9867
[259]	validation_0-rmse:9.73292	validation_1-rmse:10.9873
[260]	validation_0-rmse:9.73193	validation_1-rmse:10.987
[261]	validation_0-rmse:9.73084	validation_1-rmse:10.9862
[262]	validation_0-rmse:9.73072	validation_1-rmse:10.9862
[263]	validation_0-rmse:9.72463	validation_1-rmse:10.9821
[264]	validation_0-rmse:9.72004	validation_1-rmse:10.98
[265]	validation_0-rmse:9.71964	validation_1-rmse:10.9796
[266]	validation_0-rmse:9.71606	validation_1-rmse:10.9759
[267]	validation_0-rmse:9.71501	validation_1-rmse:10.9759
[268]	validation_0-rmse:9.71291	validation_1-rmse:10.975
[269]	validation_0-rmse:9.71127	validation_1-rmse:10.9748
[270]	validation_0-rmse:9.71108	validation_1-rmse:10.975
[271]	validation_0-rmse:9.71092	validation_1-rmse:10.9753
[272]	validation_0-rmse:9.70684	validation_1-rmse:10.9745
[273]	validation_0-rmse:9.70601	validation_1-rmse:10.9745
[274]	validation_0-rmse:9.70489	validation_1-rmse:10.9746
[275]	validation_0-rmse:9.70482	validation_1-rmse:10.9752
[276]	validation_0-rmse:9.70441	validation_1-rmse:10.9758
[277]	validation_0-rmse:9.70192	validation_1-rmse:10.9743
[278]	validation_0-rmse:9.70184	validation_1-rmse:10.9742
[279]	validation_0-rmse:9.70047	validation_1-rmse:10.9739
[280]	validation_0-rmse:9.69984	validation_1-rmse:10.9741
[281]	validation_0-rmse:9.69973	validation_1-rmse:10.9743
Stopping. Best iteration:
[231]	validation_0-rmse:9.79273	validation_1-rmse:10.9664

[0]	validation_0-rmse:24.4328	validation_1-rmse:24.7542
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.4915	validation_1-rmse:22.824
[2]	validation_0-rmse:20.7828	validation_1-rmse:21.128
[3]	validation_0-rmse:19.4006	validation_1-rmse:19.7589
[4]	validation_0-rmse:18.3702	validation_1-rmse:18.7414
[5]	validation_0-rmse:17.3344	validation_1-rmse:17.7209
[6]	validation_0-rmse:16.3638	validation_1-rmse:16.7619
[7]	validation_0-rmse:15.5934	validation_1-rmse:16.0044
[8]	validation_0-rmse:14.804	validation_1-rmse:15.2304
[9]	validation_0-rmse:14.1182	validation_1-rmse:14.56
[10]	validation_0-rmse:13.5234	validation_1-rmse:13.9837
[11]	validation_0-rmse:13.0261	validation_1-rmse:13.5004
[12]	validation_0-rmse:12.6993	validation_1-rmse:13.1843
[13]	validation_0-rmse:12.4596	validation_1-rmse:12.9553
[14]	validation_0-rmse:12.179	validation_1-rmse:12.6859
[15]	validation_0-rmse:11.8994	validation_1-rmse:12.4188
[16]	validation_0-rmse:11.7029	validation_1-rmse:12.2345
[17]	validation_0-rmse:11.496	validation_1-rmse:12.0399
[18]	validation_0-rmse:11.3541	validation_1-rmse:11.9074
[19]	validation_0-rmse:11.2001	validation_1-rmse:11.7616
[20]	validation_0-rmse:11.1135	validation_1-rmse:11.6824
[21]	validation_0-rmse:11.0208	validation_1-rmse:11.6037
[22]	validation_0-rmse:10.9688	validation_1-rmse:11.558
[23]	validation_0-rmse:10.8793	validation_1-rmse:11.4742
[24]	validation_0-rmse:10.8287	validation_1-rmse:11.4293
[25]	validation_0-rmse:10.7533	validation_1-rmse:11.3647
[26]	validation_0-rmse:10.7195	validation_1-rmse:11.337
[27]	validation_0-rmse:10.6905	validation_1-rmse:11.3121
[28]	validation_0-rmse:10.6461	validation_1-rmse:11.2833
[29]	validation_0-rmse:10.6221	validation_1-rmse:11.2633
[30]	validation_0-rmse:10.5905	validation_1-rmse:11.2397
[31]	validation_0-rmse:10.5393	validation_1-rmse:11.2012
[32]	validation_0-rmse:10.5234	validation_1-rmse:11.1892
[33]	validation_0-rmse:10.4869	validation_1-rmse:11.1596
[34]	validation_0-rmse:10.4607	validation_1-rmse:11.1416
[35]	validation_0-rmse:10.4465	validation_1-rmse:11.1334
[36]	validation_0-rmse:10.424	validation_1-rmse:11.1157
[37]	validation_0-rmse:10.4035	validation_1-rmse:11.0996
[38]	validation_0-rmse:10.3753	validation_1-rmse:11.0822
[39]	validation_0-rmse:10.3574	validation_1-rmse:11.07
[40]	validation_0-rmse:10.3284	validation_1-rmse:11.0585
[41]	validation_0-rmse:10.3233	validation_1-rmse:11.0565
[42]	validation_0-rmse:10.311	validation_1-rmse:11.0499
[43]	validation_0-rmse:10.2957	validation_1-rmse:11.0383
[44]	validation_0-rmse:10.2838	validation_1-rmse:11.0385
[45]	validation_0-rmse:10.2712	validation_1-rmse:11.0368
[46]	validation_0-rmse:10.2603	validation_1-rmse:11.0305
[47]	validation_0-rmse:10.2549	validation_1-rmse:11.027
[48]	validation_0-rmse:10.2376	validation_1-rmse:11.0161
[49]	validation_0-rmse:10.2325	validation_1-rmse:11.0153
[50]	validation_0-rmse:10.2232	validation_1-rmse:11.0189
[51]	validation_0-rmse:10.2211	validation_1-rmse:11.0185
[52]	validation_0-rmse:10.2079	validation_1-rmse:11.0109
[53]	validation_0-rmse:10.1963	validation_1-rmse:11.0032
[54]	validation_0-rmse:10.1821	validation_1-rmse:11.0043
[55]	validation_0-rmse:10.1731	validation_1-rmse:10.9987
[56]	validation_0-rmse:10.1625	validation_1-rmse:10.9948
[57]	validation_0-rmse:10.1592	validation_1-rmse:10.9948
[58]	validation_0-rmse:10.1516	validation_1-rmse:10.9913
[59]	validation_0-rmse:10.1486	validation_1-rmse:10.9891
[60]	validation_0-rmse:10.1397	validation_1-rmse:10.987
[61]	validation_0-rmse:10.1281	validation_1-rmse:10.9814
[62]	validation_0-rmse:10.1252	validation_1-rmse:10.9802
[63]	validation_0-rmse:10.1147	validation_1-rmse:10.9763
[64]	validation_0-rmse:10.1111	validation_1-rmse:10.9747
[65]	validation_0-rmse:10.108	validation_1-rmse:10.9741
[66]	validation_0-rmse:10.1016	validation_1-rmse:10.9781
[67]	validation_0-rmse:10.0895	validation_1-rmse:10.9723
[68]	validation_0-rmse:10.0847	validation_1-rmse:10.9686
[69]	validation_0-rmse:10.0817	validation_1-rmse:10.9714
[70]	validation_0-rmse:10.0722	validation_1-rmse:10.9645
[71]	validation_0-rmse:10.0669	validation_1-rmse:10.9549
[72]	validation_0-rmse:10.0639	validation_1-rmse:10.9525
[73]	validation_0-rmse:10.0621	validation_1-rmse:10.9542
[74]	validation_0-rmse:10.0492	validation_1-rmse:10.9473
[75]	validation_0-rmse:10.0382	validation_1-rmse:10.9451
[76]	validation_0-rmse:10.0251	validation_1-rmse:10.938
[77]	validation_0-rmse:10.0226	validation_1-rmse:10.9382
[78]	validation_0-rmse:10.0125	validation_1-rmse:10.9327
[79]	validation_0-rmse:10.0069	validation_1-rmse:10.9297
[80]	validation_0-rmse:10.003	validation_1-rmse:10.9277
[81]	validation_0-rmse:10.0023	validation_1-rmse:10.9277
[82]	validation_0-rmse:9.99671	validation_1-rmse:10.9288
[83]	validation_0-rmse:9.99403	validation_1-rmse:10.9264
[84]	validation_0-rmse:9.98584	validation_1-rmse:10.9224
[85]	validation_0-rmse:9.97616	validation_1-rmse:10.9162
[86]	validation_0-rmse:9.97292	validation_1-rmse:10.9162
[87]	validation_0-rmse:9.96916	validation_1-rmse:10.9153
[88]	validation_0-rmse:9.96523	validation_1-rmse:10.914
[89]	validation_0-rmse:9.96425	validation_1-rmse:10.9064
[90]	validation_0-rmse:9.96129	validation_1-rmse:10.9061
[91]	validation_0-rmse:9.9599	validation_1-rmse:10.906
[92]	validation_0-rmse:9.95828	validation_1-rmse:10.9053
[93]	validation_0-rmse:9.95381	validation_1-rmse:10.9113
[94]	validation_0-rmse:9.95197	validation_1-rmse:10.9131
[95]	validation_0-rmse:9.94519	validation_1-rmse:10.9083
[96]	validation_0-rmse:9.94201	validation_1-rmse:10.9068
[97]	validation_0-rmse:9.93321	validation_1-rmse:10.9045
[98]	validation_0-rmse:9.92606	validation_1-rmse:10.903
[99]	validation_0-rmse:9.91815	validation_1-rmse:10.8976
[100]	validation_0-rmse:9.91484	validation_1-rmse:10.9001
[101]	validation_0-rmse:9.9127	validation_1-rmse:10.9007
[102]	validation_0-rmse:9.90663	validation_1-rmse:10.8984
[103]	validation_0-rmse:9.90113	validation_1-rmse:10.8961
[104]	validation_0-rmse:9.89024	validation_1-rmse:10.901
[105]	validation_0-rmse:9.88889	validation_1-rmse:10.901
[106]	validation_0-rmse:9.8835	validation_1-rmse:10.8988
[107]	validation_0-rmse:9.87786	validation_1-rmse:10.8961
[108]	validation_0-rmse:9.87663	validation_1-rmse:10.8973
[109]	validation_0-rmse:9.87392	validation_1-rmse:10.8988
[110]	validation_0-rmse:9.87039	validation_1-rmse:10.897
[111]	validation_0-rmse:9.86797	validation_1-rmse:10.8959
[112]	validation_0-rmse:9.86247	validation_1-rmse:10.8943
[113]	validation_0-rmse:9.85913	validation_1-rmse:10.8919
[114]	validation_0-rmse:9.85754	validation_1-rmse:10.8914
[115]	validation_0-rmse:9.85476	validation_1-rmse:10.891
[116]	validation_0-rmse:9.84868	validation_1-rmse:10.889
[117]	validation_0-rmse:9.8465	validation_1-rmse:10.8895
[118]	validation_0-rmse:9.84523	validation_1-rmse:10.8893
[119]	validation_0-rmse:9.8424	validation_1-rmse:10.889
[120]	validation_0-rmse:9.84196	validation_1-rmse:10.8889
[121]	validation_0-rmse:9.84094	validation_1-rmse:10.8895
[122]	validation_0-rmse:9.83066	validation_1-rmse:10.8872
[123]	validation_0-rmse:9.82804	validation_1-rmse:10.8862
[124]	validation_0-rmse:9.8261	validation_1-rmse:10.8846
[125]	validation_0-rmse:9.82176	validation_1-rmse:10.883
[126]	validation_0-rmse:9.82077	validation_1-rmse:10.8854
[127]	validation_0-rmse:9.81804	validation_1-rmse:10.8842
[128]	validation_0-rmse:9.8133	validation_1-rmse:10.8827
[129]	validation_0-rmse:9.81173	validation_1-rmse:10.8812
[130]	validation_0-rmse:9.81167	validation_1-rmse:10.882
[131]	validation_0-rmse:9.80623	validation_1-rmse:10.8826
[132]	validation_0-rmse:9.8059	validation_1-rmse:10.8835
[133]	validation_0-rmse:9.80439	validation_1-rmse:10.8833
[134]	validation_0-rmse:9.80181	validation_1-rmse:10.8868
[135]	validation_0-rmse:9.79721	validation_1-rmse:10.8862
[136]	validation_0-rmse:9.79289	validation_1-rmse:10.8846
[137]	validation_0-rmse:9.79275	validation_1-rmse:10.8846
[138]	validation_0-rmse:9.79198	validation_1-rmse:10.8852
[139]	validation_0-rmse:9.78993	validation_1-rmse:10.8842
[140]	validation_0-rmse:9.78735	validation_1-rmse:10.8846
[141]	validation_0-rmse:9.78565	validation_1-rmse:10.8848
[142]	validation_0-rmse:9.77864	validation_1-rmse:10.8861
[143]	validation_0-rmse:9.77741	validation_1-rmse:10.886
[144]	validation_0-rmse:9.77437	validation_1-rmse:10.8831
[145]	validation_0-rmse:9.77378	validation_1-rmse:10.8842
[146]	validation_0-rmse:9.77346	validation_1-rmse:10.8838
[147]	validation_0-rmse:9.77175	validation_1-rmse:10.884
[148]	validation_0-rmse:9.76959	validation_1-rmse:10.8836
[149]	validation_0-rmse:9.76749	validation_1-rmse:10.883
[150]	validation_0-rmse:9.76598	validation_1-rmse:10.8818
[151]	validation_0-rmse:9.76453	validation_1-rmse:10.8814
[152]	validation_0-rmse:9.76008	validation_1-rmse:10.8805
[153]	validation_0-rmse:9.7552	validation_1-rmse:10.8818
[154]	validation_0-rmse:9.74733	validation_1-rmse:10.8835
[155]	validation_0-rmse:9.74194	validation_1-rmse:10.8805
[156]	validation_0-rmse:9.74084	validation_1-rmse:10.8809
[157]	validation_0-rmse:9.73773	validation_1-rmse:10.8815
[158]	validation_0-rmse:9.73272	validation_1-rmse:10.8787
[159]	validation_0-rmse:9.73232	validation_1-rmse:10.8795
[160]	validation_0-rmse:9.73043	validation_1-rmse:10.8783
[161]	validation_0-rmse:9.73007	validation_1-rmse:10.8793
[162]	validation_0-rmse:9.72584	validation_1-rmse:10.8743
[163]	validation_0-rmse:9.72259	validation_1-rmse:10.8707
[164]	validation_0-rmse:9.71744	validation_1-rmse:10.8729
[165]	validation_0-rmse:9.71544	validation_1-rmse:10.8728
[166]	validation_0-rmse:9.71461	validation_1-rmse:10.8733
[167]	validation_0-rmse:9.7144	validation_1-rmse:10.8732
[168]	validation_0-rmse:9.71376	validation_1-rmse:10.8732
[169]	validation_0-rmse:9.71204	validation_1-rmse:10.873
[170]	validation_0-rmse:9.71116	validation_1-rmse:10.8717
[171]	validation_0-rmse:9.70721	validation_1-rmse:10.8723
[172]	validation_0-rmse:9.7063	validation_1-rmse:10.8743
[173]	validation_0-rmse:9.69338	validation_1-rmse:10.8785
[174]	validation_0-rmse:9.69254	validation_1-rmse:10.8785
[175]	validation_0-rmse:9.68921	validation_1-rmse:10.8771
[176]	validation_0-rmse:9.68602	validation_1-rmse:10.8774
[177]	validation_0-rmse:9.68074	validation_1-rmse:10.8757
[178]	validation_0-rmse:9.67709	validation_1-rmse:10.8755
[179]	validation_0-rmse:9.67675	validation_1-rmse:10.876
[180]	validation_0-rmse:9.67566	validation_1-rmse:10.8775
[181]	validation_0-rmse:9.67167	validation_1-rmse:10.8756
[182]	validation_0-rmse:9.67053	validation_1-rmse:10.8756
[183]	validation_0-rmse:9.66968	validation_1-rmse:10.8757
[184]	validation_0-rmse:9.66958	validation_1-rmse:10.8756
[185]	validation_0-rmse:9.6673	validation_1-rmse:10.8753
[186]	validation_0-rmse:9.6664	validation_1-rmse:10.8755
[187]	validation_0-rmse:9.66596	validation_1-rmse:10.875
[188]	validation_0-rmse:9.66433	validation_1-rmse:10.8766
[189]	validation_0-rmse:9.66264	validation_1-rmse:10.8749
[190]	validation_0-rmse:9.66251	validation_1-rmse:10.8749
[191]	validation_0-rmse:9.66078	validation_1-rmse:10.8767
[192]	validation_0-rmse:9.65815	validation_1-rmse:10.8754
[193]	validation_0-rmse:9.65678	validation_1-rmse:10.8781
[194]	validation_0-rmse:9.65251	validation_1-rmse:10.8768
[195]	validation_0-rmse:9.64946	validation_1-rmse:10.8773
[196]	validation_0-rmse:9.64915	validation_1-rmse:10.8788
[197]	validation_0-rmse:9.6466	validation_1-rmse:10.8775
[198]	validation_0-rmse:9.64034	validation_1-rmse:10.8749
[199]	validation_0-rmse:9.64026	validation_1-rmse:10.8731
[200]	validation_0-rmse:9.63655	validation_1-rmse:10.8694
[201]	validation_0-rmse:9.63521	validation_1-rmse:10.8702
[202]	validation_0-rmse:9.6298	validation_1-rmse:10.8721
[203]	validation_0-rmse:9.62949	validation_1-rmse:10.873
[204]	validation_0-rmse:9.62699	validation_1-rmse:10.8713
[205]	validation_0-rmse:9.62236	validation_1-rmse:10.8695
[206]	validation_0-rmse:9.62204	validation_1-rmse:10.8706
[207]	validation_0-rmse:9.61838	validation_1-rmse:10.87
[208]	validation_0-rmse:9.61692	validation_1-rmse:10.8702
[209]	validation_0-rmse:9.61524	validation_1-rmse:10.8695
[210]	validation_0-rmse:9.61058	validation_1-rmse:10.8669
[211]	validation_0-rmse:9.60873	validation_1-rmse:10.8656
[212]	validation_0-rmse:9.60812	validation_1-rmse:10.8656
[213]	validation_0-rmse:9.60551	validation_1-rmse:10.8646
[214]	validation_0-rmse:9.60528	validation_1-rmse:10.8676
[215]	validation_0-rmse:9.60062	validation_1-rmse:10.8668
[216]	validation_0-rmse:9.59995	validation_1-rmse:10.8658
[217]	validation_0-rmse:9.59784	validation_1-rmse:10.8649
[218]	validation_0-rmse:9.59744	validation_1-rmse:10.8644
[219]	validation_0-rmse:9.59655	validation_1-rmse:10.8653
[220]	validation_0-rmse:9.58853	validation_1-rmse:10.8615
[221]	validation_0-rmse:9.58749	validation_1-rmse:10.8615
[222]	validation_0-rmse:9.58429	validation_1-rmse:10.8601
[223]	validation_0-rmse:9.58199	validation_1-rmse:10.865
[224]	validation_0-rmse:9.58079	validation_1-rmse:10.8618
[225]	validation_0-rmse:9.58047	validation_1-rmse:10.8618
[226]	validation_0-rmse:9.58037	validation_1-rmse:10.8617
[227]	validation_0-rmse:9.57908	validation_1-rmse:10.8617
[228]	validation_0-rmse:9.57782	validation_1-rmse:10.8624
[229]	validation_0-rmse:9.5774	validation_1-rmse:10.8637
[230]	validation_0-rmse:9.57209	validation_1-rmse:10.8691
[231]	validation_0-rmse:9.56995	validation_1-rmse:10.8692
[232]	validation_0-rmse:9.56486	validation_1-rmse:10.8706
[233]	validation_0-rmse:9.56026	validation_1-rmse:10.8683
[234]	validation_0-rmse:9.55339	validation_1-rmse:10.8676
[235]	validation_0-rmse:9.54827	validation_1-rmse:10.8646
[236]	validation_0-rmse:9.54779	validation_1-rmse:10.8653
[237]	validation_0-rmse:9.54572	validation_1-rmse:10.8653
[238]	validation_0-rmse:9.54294	validation_1-rmse:10.8656
[239]	validation_0-rmse:9.54265	validation_1-rmse:10.8656
[240]	validation_0-rmse:9.54155	validation_1-rmse:10.8654
[241]	validation_0-rmse:9.54085	validation_1-rmse:10.8654
[242]	validation_0-rmse:9.53714	validation_1-rmse:10.8709
[243]	validation_0-rmse:9.53587	validation_1-rmse:10.8707
[244]	validation_0-rmse:9.53268	validation_1-rmse:10.8703
[245]	validation_0-rmse:9.5302	validation_1-rmse:10.87
[246]	validation_0-rmse:9.52961	validation_1-rmse:10.8701
[247]	validation_0-rmse:9.52771	validation_1-rmse:10.8657
[248]	validation_0-rmse:9.52692	validation_1-rmse:10.8667
[249]	validation_0-rmse:9.5267	validation_1-rmse:10.8678
[250]	validation_0-rmse:9.52663	validation_1-rmse:10.8678
[251]	validation_0-rmse:9.52623	validation_1-rmse:10.8677
[252]	validation_0-rmse:9.52583	validation_1-rmse:10.8691
[253]	validation_0-rmse:9.52523	validation_1-rmse:10.8688
[254]	validation_0-rmse:9.52508	validation_1-rmse:10.8693
[255]	validation_0-rmse:9.525	validation_1-rmse:10.8693
[256]	validation_0-rmse:9.52196	validation_1-rmse:10.8673
[257]	validation_0-rmse:9.52092	validation_1-rmse:10.868
[258]	validation_0-rmse:9.52052	validation_1-rmse:10.868
[259]	validation_0-rmse:9.52007	validation_1-rmse:10.8689
[260]	validation_0-rmse:9.51765	validation_1-rmse:10.8678
[261]	validation_0-rmse:9.51623	validation_1-rmse:10.8677
[262]	validation_0-rmse:9.51617	validation_1-rmse:10.8677
[263]	validation_0-rmse:9.51356	validation_1-rmse:10.8672
[264]	validation_0-rmse:9.5127	validation_1-rmse:10.8692
[265]	validation_0-rmse:9.51051	validation_1-rmse:10.8677
[266]	validation_0-rmse:9.50845	validation_1-rmse:10.8669
[267]	validation_0-rmse:9.50604	validation_1-rmse:10.8673
[268]	validation_0-rmse:9.50541	validation_1-rmse:10.8679
[269]	validation_0-rmse:9.50173	validation_1-rmse:10.8673
[270]	validation_0-rmse:9.50163	validation_1-rmse:10.8676
[271]	validation_0-rmse:9.50146	validation_1-rmse:10.8676
[272]	validation_0-rmse:9.49912	validation_1-rmse:10.8651
Stopping. Best iteration:
[222]	validation_0-rmse:9.58429	validation_1-rmse:10.8601

[0]	validation_0-rmse:24.2273	validation_1-rmse:24.5412
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.316	validation_1-rmse:22.639
[2]	validation_0-rmse:20.6353	validation_1-rmse:20.9659
[3]	validation_0-rmse:19.1663	validation_1-rmse:19.5084
[4]	validation_0-rmse:17.8856	validation_1-rmse:18.2384
[5]	validation_0-rmse:16.7708	validation_1-rmse:17.1332
[6]	validation_0-rmse:15.8069	validation_1-rmse:16.1851
[7]	validation_0-rmse:14.9761	validation_1-rmse:15.3668
[8]	validation_0-rmse:14.2686	validation_1-rmse:14.6694
[9]	validation_0-rmse:13.6661	validation_1-rmse:14.0777
[10]	validation_0-rmse:13.1518	validation_1-rmse:13.5746
[11]	validation_0-rmse:12.7212	validation_1-rmse:13.1538
[12]	validation_0-rmse:12.3584	validation_1-rmse:12.8029
[13]	validation_0-rmse:12.0496	validation_1-rmse:12.5091
[14]	validation_0-rmse:11.7963	validation_1-rmse:12.2618
[15]	validation_0-rmse:11.585	validation_1-rmse:12.0614
[16]	validation_0-rmse:11.4044	validation_1-rmse:11.892
[17]	validation_0-rmse:11.2575	validation_1-rmse:11.7513
[18]	validation_0-rmse:11.1353	validation_1-rmse:11.6328
[19]	validation_0-rmse:11.0344	validation_1-rmse:11.5359
[20]	validation_0-rmse:10.9371	validation_1-rmse:11.4544
[21]	validation_0-rmse:10.8677	validation_1-rmse:11.389
[22]	validation_0-rmse:10.8057	validation_1-rmse:11.3302
[23]	validation_0-rmse:10.7517	validation_1-rmse:11.2825
[24]	validation_0-rmse:10.7046	validation_1-rmse:11.241
[25]	validation_0-rmse:10.6675	validation_1-rmse:11.2059
[26]	validation_0-rmse:10.6355	validation_1-rmse:11.1786
[27]	validation_0-rmse:10.6059	validation_1-rmse:11.1601
[28]	validation_0-rmse:10.5625	validation_1-rmse:11.1421
[29]	validation_0-rmse:10.543	validation_1-rmse:11.1243
[30]	validation_0-rmse:10.5153	validation_1-rmse:11.1097
[31]	validation_0-rmse:10.4922	validation_1-rmse:11.1078
[32]	validation_0-rmse:10.4782	validation_1-rmse:11.0975
[33]	validation_0-rmse:10.4658	validation_1-rmse:11.0869
[34]	validation_0-rmse:10.4446	validation_1-rmse:11.0785
[35]	validation_0-rmse:10.4272	validation_1-rmse:11.0713
[36]	validation_0-rmse:10.4164	validation_1-rmse:11.063
[37]	validation_0-rmse:10.4101	validation_1-rmse:11.0581
[38]	validation_0-rmse:10.4013	validation_1-rmse:11.0525
[39]	validation_0-rmse:10.3898	validation_1-rmse:11.0466
[40]	validation_0-rmse:10.3693	validation_1-rmse:11.0457
[41]	validation_0-rmse:10.3583	validation_1-rmse:11.0435
[42]	validation_0-rmse:10.3491	validation_1-rmse:11.0425
[43]	validation_0-rmse:10.3422	validation_1-rmse:11.0376
[44]	validation_0-rmse:10.3308	validation_1-rmse:11.037
[45]	validation_0-rmse:10.319	validation_1-rmse:11.0379
[46]	validation_0-rmse:10.3089	validation_1-rmse:11.0365
[47]	validation_0-rmse:10.3033	validation_1-rmse:11.033
[48]	validation_0-rmse:10.2937	validation_1-rmse:11.0304
[49]	validation_0-rmse:10.2888	validation_1-rmse:11.028
[50]	validation_0-rmse:10.2821	validation_1-rmse:11.0308
[51]	validation_0-rmse:10.272	validation_1-rmse:11.0398
[52]	validation_0-rmse:10.2674	validation_1-rmse:11.0372
[53]	validation_0-rmse:10.2644	validation_1-rmse:11.0352
[54]	validation_0-rmse:10.2536	validation_1-rmse:11.0324
[55]	validation_0-rmse:10.2399	validation_1-rmse:11.0399
[56]	validation_0-rmse:10.2329	validation_1-rmse:11.0352
[57]	validation_0-rmse:10.228	validation_1-rmse:11.0354
[58]	validation_0-rmse:10.2214	validation_1-rmse:11.0313
[59]	validation_0-rmse:10.2089	validation_1-rmse:11.0264
[60]	validation_0-rmse:10.201	validation_1-rmse:11.0246
[61]	validation_0-rmse:10.1936	validation_1-rmse:11.0268
[62]	validation_0-rmse:10.1888	validation_1-rmse:11.0253
[63]	validation_0-rmse:10.1778	validation_1-rmse:11.0341
[64]	validation_0-rmse:10.172	validation_1-rmse:11.0368
[65]	validation_0-rmse:10.167	validation_1-rmse:11.0353
[66]	validation_0-rmse:10.161	validation_1-rmse:11.0352
[67]	validation_0-rmse:10.1561	validation_1-rmse:11.0304
[68]	validation_0-rmse:10.1493	validation_1-rmse:11.029
[69]	validation_0-rmse:10.1409	validation_1-rmse:11.027
[70]	validation_0-rmse:10.1285	validation_1-rmse:11.0318
[71]	validation_0-rmse:10.1205	validation_1-rmse:11.0364
[72]	validation_0-rmse:10.1118	validation_1-rmse:11.0425
[73]	validation_0-rmse:10.1083	validation_1-rmse:11.0429
[74]	validation_0-rmse:10.1003	validation_1-rmse:11.0385
[75]	validation_0-rmse:10.0958	validation_1-rmse:11.037
[76]	validation_0-rmse:10.0891	validation_1-rmse:11.0333
[77]	validation_0-rmse:10.0826	validation_1-rmse:11.0352
[78]	validation_0-rmse:10.078	validation_1-rmse:11.0365
[79]	validation_0-rmse:10.0698	validation_1-rmse:11.0356
[80]	validation_0-rmse:10.0664	validation_1-rmse:11.0352
[81]	validation_0-rmse:10.0555	validation_1-rmse:11.0301
[82]	validation_0-rmse:10.0533	validation_1-rmse:11.0311
[83]	validation_0-rmse:10.046	validation_1-rmse:11.0399
[84]	validation_0-rmse:10.0392	validation_1-rmse:11.0406
[85]	validation_0-rmse:10.0304	validation_1-rmse:11.0414
[86]	validation_0-rmse:10.0211	validation_1-rmse:11.0492
[87]	validation_0-rmse:10.019	validation_1-rmse:11.0502
[88]	validation_0-rmse:10.0129	validation_1-rmse:11.0512
[89]	validation_0-rmse:10.0093	validation_1-rmse:11.0521
[90]	validation_0-rmse:10.0032	validation_1-rmse:11.0476
[91]	validation_0-rmse:9.99545	validation_1-rmse:11.055
[92]	validation_0-rmse:9.99261	validation_1-rmse:11.0543
[93]	validation_0-rmse:9.98949	validation_1-rmse:11.0549
[94]	validation_0-rmse:9.98268	validation_1-rmse:11.0522
[95]	validation_0-rmse:9.97924	validation_1-rmse:11.0513
[96]	validation_0-rmse:9.97618	validation_1-rmse:11.0513
[97]	validation_0-rmse:9.97181	validation_1-rmse:11.0532
[98]	validation_0-rmse:9.96386	validation_1-rmse:11.0568
[99]	validation_0-rmse:9.95634	validation_1-rmse:11.0683
[100]	validation_0-rmse:9.95178	validation_1-rmse:11.0765
[101]	validation_0-rmse:9.94613	validation_1-rmse:11.0869
[102]	validation_0-rmse:9.94272	validation_1-rmse:11.0887
[103]	validation_0-rmse:9.93924	validation_1-rmse:11.088
[104]	validation_0-rmse:9.93631	validation_1-rmse:11.0913
[105]	validation_0-rmse:9.93327	validation_1-rmse:11.0907
[106]	validation_0-rmse:9.93076	validation_1-rmse:11.0947
[107]	validation_0-rmse:9.92203	validation_1-rmse:11.0938
[108]	validation_0-rmse:9.92126	validation_1-rmse:11.094
[109]	validation_0-rmse:9.91831	validation_1-rmse:11.0928
[110]	validation_0-rmse:9.91537	validation_1-rmse:11.0911
Stopping. Best iteration:
[60]	validation_0-rmse:10.201	validation_1-rmse:11.0246

[0]	validation_0-rmse:24.2388	validation_1-rmse:24.5521
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3307	validation_1-rmse:22.651
[2]	validation_0-rmse:20.6512	validation_1-rmse:20.9775
[3]	validation_0-rmse:19.1861	validation_1-rmse:19.5191
[4]	validation_0-rmse:17.9103	validation_1-rmse:18.2499
[5]	validation_0-rmse:16.7998	validation_1-rmse:17.1462
[6]	validation_0-rmse:15.8409	validation_1-rmse:16.196
[7]	validation_0-rmse:15.0159	validation_1-rmse:15.3806
[8]	validation_0-rmse:14.313	validation_1-rmse:14.6891
[9]	validation_0-rmse:13.716	validation_1-rmse:14.1004
[10]	validation_0-rmse:13.2071	validation_1-rmse:13.5969
[11]	validation_0-rmse:12.7818	validation_1-rmse:13.1774
[12]	validation_0-rmse:12.4223	validation_1-rmse:12.8263
[13]	validation_0-rmse:12.1217	validation_1-rmse:12.5315
[14]	validation_0-rmse:11.8733	validation_1-rmse:12.2898
[15]	validation_0-rmse:11.6643	validation_1-rmse:12.0894
[16]	validation_0-rmse:11.4881	validation_1-rmse:11.9222
[17]	validation_0-rmse:11.3415	validation_1-rmse:11.7815
[18]	validation_0-rmse:11.2219	validation_1-rmse:11.6648
[19]	validation_0-rmse:11.1232	validation_1-rmse:11.5701
[20]	validation_0-rmse:11.0397	validation_1-rmse:11.4893
[21]	validation_0-rmse:10.9717	validation_1-rmse:11.4252
[22]	validation_0-rmse:10.914	validation_1-rmse:11.3693
[23]	validation_0-rmse:10.8636	validation_1-rmse:11.3242
[24]	validation_0-rmse:10.8209	validation_1-rmse:11.2819
[25]	validation_0-rmse:10.7874	validation_1-rmse:11.2506
[26]	validation_0-rmse:10.7585	validation_1-rmse:11.2222
[27]	validation_0-rmse:10.7305	validation_1-rmse:11.1992
[28]	validation_0-rmse:10.7015	validation_1-rmse:11.1804
[29]	validation_0-rmse:10.6821	validation_1-rmse:11.1646
[30]	validation_0-rmse:10.6595	validation_1-rmse:11.1501
[31]	validation_0-rmse:10.6426	validation_1-rmse:11.1428
[32]	validation_0-rmse:10.6263	validation_1-rmse:11.1308
[33]	validation_0-rmse:10.6153	validation_1-rmse:11.122
[34]	validation_0-rmse:10.5943	validation_1-rmse:11.1129
[35]	validation_0-rmse:10.5834	validation_1-rmse:11.1044
[36]	validation_0-rmse:10.5747	validation_1-rmse:11.0966
[37]	validation_0-rmse:10.5693	validation_1-rmse:11.0929
[38]	validation_0-rmse:10.5635	validation_1-rmse:11.0911
[39]	validation_0-rmse:10.5563	validation_1-rmse:11.0868
[40]	validation_0-rmse:10.5484	validation_1-rmse:11.0836
[41]	validation_0-rmse:10.5444	validation_1-rmse:11.0819
[42]	validation_0-rmse:10.5366	validation_1-rmse:11.0785
[43]	validation_0-rmse:10.5304	validation_1-rmse:11.0736
[44]	validation_0-rmse:10.5209	validation_1-rmse:11.0738
[45]	validation_0-rmse:10.5156	validation_1-rmse:11.0711
[46]	validation_0-rmse:10.51	validation_1-rmse:11.0689
[47]	validation_0-rmse:10.5059	validation_1-rmse:11.0659
[48]	validation_0-rmse:10.5021	validation_1-rmse:11.0659
[49]	validation_0-rmse:10.498	validation_1-rmse:11.0622
[50]	validation_0-rmse:10.4912	validation_1-rmse:11.0635
[51]	validation_0-rmse:10.4744	validation_1-rmse:11.0676
[52]	validation_0-rmse:10.4707	validation_1-rmse:11.0669
[53]	validation_0-rmse:10.4699	validation_1-rmse:11.0664
[54]	validation_0-rmse:10.4628	validation_1-rmse:11.0639
[55]	validation_0-rmse:10.4531	validation_1-rmse:11.0677
[56]	validation_0-rmse:10.45	validation_1-rmse:11.0703
[57]	validation_0-rmse:10.4473	validation_1-rmse:11.0686
[58]	validation_0-rmse:10.4436	validation_1-rmse:11.0662
[59]	validation_0-rmse:10.4365	validation_1-rmse:11.0638
[60]	validation_0-rmse:10.4314	validation_1-rmse:11.0628
[61]	validation_0-rmse:10.4223	validation_1-rmse:11.0654
[62]	validation_0-rmse:10.4172	validation_1-rmse:11.06
[63]	validation_0-rmse:10.409	validation_1-rmse:11.0569
[64]	validation_0-rmse:10.4004	validation_1-rmse:11.0576
[65]	validation_0-rmse:10.3934	validation_1-rmse:11.0526
[66]	validation_0-rmse:10.3917	validation_1-rmse:11.0511
[67]	validation_0-rmse:10.385	validation_1-rmse:11.0521
[68]	validation_0-rmse:10.3803	validation_1-rmse:11.048
[69]	validation_0-rmse:10.3748	validation_1-rmse:11.0506
[70]	validation_0-rmse:10.364	validation_1-rmse:11.0502
[71]	validation_0-rmse:10.3614	validation_1-rmse:11.0526
[72]	validation_0-rmse:10.3544	validation_1-rmse:11.061
[73]	validation_0-rmse:10.349	validation_1-rmse:11.0622
[74]	validation_0-rmse:10.348	validation_1-rmse:11.0638
[75]	validation_0-rmse:10.3443	validation_1-rmse:11.0603
[76]	validation_0-rmse:10.3411	validation_1-rmse:11.0595
[77]	validation_0-rmse:10.3371	validation_1-rmse:11.0607
[78]	validation_0-rmse:10.3301	validation_1-rmse:11.0583
[79]	validation_0-rmse:10.3248	validation_1-rmse:11.0539
[80]	validation_0-rmse:10.324	validation_1-rmse:11.054
[81]	validation_0-rmse:10.3161	validation_1-rmse:11.0467
[82]	validation_0-rmse:10.3145	validation_1-rmse:11.0486
[83]	validation_0-rmse:10.3103	validation_1-rmse:11.0503
[84]	validation_0-rmse:10.3073	validation_1-rmse:11.0484
[85]	validation_0-rmse:10.3013	validation_1-rmse:11.0532
[86]	validation_0-rmse:10.2997	validation_1-rmse:11.0516
[87]	validation_0-rmse:10.2995	validation_1-rmse:11.0511
[88]	validation_0-rmse:10.2978	validation_1-rmse:11.0492
[89]	validation_0-rmse:10.297	validation_1-rmse:11.0492
[90]	validation_0-rmse:10.2941	validation_1-rmse:11.0471
[91]	validation_0-rmse:10.2895	validation_1-rmse:11.0459
[92]	validation_0-rmse:10.2876	validation_1-rmse:11.045
[93]	validation_0-rmse:10.2832	validation_1-rmse:11.0466
[94]	validation_0-rmse:10.2795	validation_1-rmse:11.0512
[95]	validation_0-rmse:10.2751	validation_1-rmse:11.0483
[96]	validation_0-rmse:10.2731	validation_1-rmse:11.0453
[97]	validation_0-rmse:10.2728	validation_1-rmse:11.0444
[98]	validation_0-rmse:10.2669	validation_1-rmse:11.0463
[99]	validation_0-rmse:10.2618	validation_1-rmse:11.0548
[100]	validation_0-rmse:10.2576	validation_1-rmse:11.0608
[101]	validation_0-rmse:10.2526	validation_1-rmse:11.0639
[102]	validation_0-rmse:10.2498	validation_1-rmse:11.0631
[103]	validation_0-rmse:10.2474	validation_1-rmse:11.0617
[104]	validation_0-rmse:10.2422	validation_1-rmse:11.064
[105]	validation_0-rmse:10.2393	validation_1-rmse:11.0637
[106]	validation_0-rmse:10.2359	validation_1-rmse:11.0683
[107]	validation_0-rmse:10.2332	validation_1-rmse:11.0678
[108]	validation_0-rmse:10.2326	validation_1-rmse:11.0671
[109]	validation_0-rmse:10.2312	validation_1-rmse:11.0633
[110]	validation_0-rmse:10.2292	validation_1-rmse:11.0618
[111]	validation_0-rmse:10.2292	validation_1-rmse:11.0624
[112]	validation_0-rmse:10.2231	validation_1-rmse:11.0611
[113]	validation_0-rmse:10.2196	validation_1-rmse:11.0603
[114]	validation_0-rmse:10.2166	validation_1-rmse:11.0579
[115]	validation_0-rmse:10.2143	validation_1-rmse:11.0573
[116]	validation_0-rmse:10.2127	validation_1-rmse:11.056
[117]	validation_0-rmse:10.2073	validation_1-rmse:11.0548
[118]	validation_0-rmse:10.2054	validation_1-rmse:11.054
[119]	validation_0-rmse:10.2046	validation_1-rmse:11.0532
[120]	validation_0-rmse:10.2008	validation_1-rmse:11.0541
[121]	validation_0-rmse:10.199	validation_1-rmse:11.0528
[122]	validation_0-rmse:10.195	validation_1-rmse:11.0479
[123]	validation_0-rmse:10.1919	validation_1-rmse:11.0472
[124]	validation_0-rmse:10.1874	validation_1-rmse:11.0472
[125]	validation_0-rmse:10.1861	validation_1-rmse:11.0471
[126]	validation_0-rmse:10.1856	validation_1-rmse:11.047
[127]	validation_0-rmse:10.1844	validation_1-rmse:11.0454
[128]	validation_0-rmse:10.1826	validation_1-rmse:11.0391
[129]	validation_0-rmse:10.1822	validation_1-rmse:11.0363
[130]	validation_0-rmse:10.1787	validation_1-rmse:11.0358
[131]	validation_0-rmse:10.1735	validation_1-rmse:11.0427
[132]	validation_0-rmse:10.1699	validation_1-rmse:11.0481
[133]	validation_0-rmse:10.1664	validation_1-rmse:11.0534
[134]	validation_0-rmse:10.1626	validation_1-rmse:11.0549
[135]	validation_0-rmse:10.1605	validation_1-rmse:11.0543
[136]	validation_0-rmse:10.1594	validation_1-rmse:11.0535
[137]	validation_0-rmse:10.157	validation_1-rmse:11.059
[138]	validation_0-rmse:10.1556	validation_1-rmse:11.0578
[139]	validation_0-rmse:10.1544	validation_1-rmse:11.0535
[140]	validation_0-rmse:10.1494	validation_1-rmse:11.0596
[141]	validation_0-rmse:10.1466	validation_1-rmse:11.06
[142]	validation_0-rmse:10.1434	validation_1-rmse:11.0588
[143]	validation_0-rmse:10.1407	validation_1-rmse:11.0602
[144]	validation_0-rmse:10.1371	validation_1-rmse:11.0645
[145]	validation_0-rmse:10.1325	validation_1-rmse:11.0678
[146]	validation_0-rmse:10.1296	validation_1-rmse:11.068
[147]	validation_0-rmse:10.1277	validation_1-rmse:11.0673
[148]	validation_0-rmse:10.1249	validation_1-rmse:11.0688
[149]	validation_0-rmse:10.124	validation_1-rmse:11.0634
[150]	validation_0-rmse:10.1232	validation_1-rmse:11.063
[151]	validation_0-rmse:10.1223	validation_1-rmse:11.0632
[152]	validation_0-rmse:10.1193	validation_1-rmse:11.0706
[153]	validation_0-rmse:10.1174	validation_1-rmse:11.07
[154]	validation_0-rmse:10.1141	validation_1-rmse:11.0717
[155]	validation_0-rmse:10.1096	validation_1-rmse:11.0722
[156]	validation_0-rmse:10.1067	validation_1-rmse:11.0711
[157]	validation_0-rmse:10.1037	validation_1-rmse:11.0697
[158]	validation_0-rmse:10.103	validation_1-rmse:11.0697
[159]	validation_0-rmse:10.1003	validation_1-rmse:11.0692
[160]	validation_0-rmse:10.0979	validation_1-rmse:11.0714
[161]	validation_0-rmse:10.0961	validation_1-rmse:11.0716
[162]	validation_0-rmse:10.093	validation_1-rmse:11.0718
[163]	validation_0-rmse:10.0907	validation_1-rmse:11.0671
[164]	validation_0-rmse:10.0847	validation_1-rmse:11.0672
[165]	validation_0-rmse:10.0808	validation_1-rmse:11.0735
[166]	validation_0-rmse:10.078	validation_1-rmse:11.073
[167]	validation_0-rmse:10.0759	validation_1-rmse:11.0718
[168]	validation_0-rmse:10.0737	validation_1-rmse:11.0718
[169]	validation_0-rmse:10.072	validation_1-rmse:11.0693
[170]	validation_0-rmse:10.0713	validation_1-rmse:11.0689
[171]	validation_0-rmse:10.0631	validation_1-rmse:11.069
[172]	validation_0-rmse:10.0626	validation_1-rmse:11.07
[173]	validation_0-rmse:10.0595	validation_1-rmse:11.0713
[174]	validation_0-rmse:10.0578	validation_1-rmse:11.0716
[175]	validation_0-rmse:10.0546	validation_1-rmse:11.0752
[176]	validation_0-rmse:10.054	validation_1-rmse:11.0767
[177]	validation_0-rmse:10.05	validation_1-rmse:11.0784
[178]	validation_0-rmse:10.0468	validation_1-rmse:11.0793
[179]	validation_0-rmse:10.0455	validation_1-rmse:11.0791
[180]	validation_0-rmse:10.0452	validation_1-rmse:11.0763
Stopping. Best iteration:
[130]	validation_0-rmse:10.1787	validation_1-rmse:11.0358

[0]	validation_0-rmse:24.2186	validation_1-rmse:24.5406
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.2923	validation_1-rmse:22.6274
[2]	validation_0-rmse:20.5972	validation_1-rmse:20.9478
[3]	validation_0-rmse:19.1103	validation_1-rmse:19.4785
[4]	validation_0-rmse:17.8128	validation_1-rmse:18.204
[5]	validation_0-rmse:16.6797	validation_1-rmse:17.0912
[6]	validation_0-rmse:15.6983	validation_1-rmse:16.1293
[7]	validation_0-rmse:14.8537	validation_1-rmse:15.3055
[8]	validation_0-rmse:14.1348	validation_1-rmse:14.6096
[9]	validation_0-rmse:13.5136	validation_1-rmse:14.0113
[10]	validation_0-rmse:12.9872	validation_1-rmse:13.5015
[11]	validation_0-rmse:12.5437	validation_1-rmse:13.0785
[12]	validation_0-rmse:12.1625	validation_1-rmse:12.7198
[13]	validation_0-rmse:11.839	validation_1-rmse:12.423
[14]	validation_0-rmse:11.5742	validation_1-rmse:12.1751
[15]	validation_0-rmse:11.3537	validation_1-rmse:11.9699
[16]	validation_0-rmse:11.1601	validation_1-rmse:11.796
[17]	validation_0-rmse:11.0065	validation_1-rmse:11.6534
[18]	validation_0-rmse:10.8714	validation_1-rmse:11.5326
[19]	validation_0-rmse:10.7586	validation_1-rmse:11.4285
[20]	validation_0-rmse:10.6632	validation_1-rmse:11.3474
[21]	validation_0-rmse:10.5727	validation_1-rmse:11.2828
[22]	validation_0-rmse:10.503	validation_1-rmse:11.226
[23]	validation_0-rmse:10.4407	validation_1-rmse:11.1777
[24]	validation_0-rmse:10.3857	validation_1-rmse:11.1372
[25]	validation_0-rmse:10.3271	validation_1-rmse:11.1018
[26]	validation_0-rmse:10.2866	validation_1-rmse:11.0705
[27]	validation_0-rmse:10.2456	validation_1-rmse:11.0482
[28]	validation_0-rmse:10.2153	validation_1-rmse:11.028
[29]	validation_0-rmse:10.1904	validation_1-rmse:11.0091
[30]	validation_0-rmse:10.159	validation_1-rmse:10.9921
[31]	validation_0-rmse:10.1427	validation_1-rmse:10.9808
[32]	validation_0-rmse:10.114	validation_1-rmse:10.9672
[33]	validation_0-rmse:10.0917	validation_1-rmse:10.956
[34]	validation_0-rmse:10.0737	validation_1-rmse:10.9468
[35]	validation_0-rmse:10.0576	validation_1-rmse:10.9387
[36]	validation_0-rmse:10.0445	validation_1-rmse:10.9314
[37]	validation_0-rmse:10.0368	validation_1-rmse:10.9266
[38]	validation_0-rmse:10.0227	validation_1-rmse:10.9191
[39]	validation_0-rmse:10.0096	validation_1-rmse:10.9146
[40]	validation_0-rmse:9.99676	validation_1-rmse:10.9118
[41]	validation_0-rmse:9.98653	validation_1-rmse:10.908
[42]	validation_0-rmse:9.9694	validation_1-rmse:10.9052
[43]	validation_0-rmse:9.96298	validation_1-rmse:10.9023
[44]	validation_0-rmse:9.9533	validation_1-rmse:10.9007
[45]	validation_0-rmse:9.92712	validation_1-rmse:10.8955
[46]	validation_0-rmse:9.91908	validation_1-rmse:10.8927
[47]	validation_0-rmse:9.90666	validation_1-rmse:10.8863
[48]	validation_0-rmse:9.88014	validation_1-rmse:10.8776
[49]	validation_0-rmse:9.86865	validation_1-rmse:10.8732
[50]	validation_0-rmse:9.86148	validation_1-rmse:10.871
[51]	validation_0-rmse:9.85462	validation_1-rmse:10.8682
[52]	validation_0-rmse:9.84423	validation_1-rmse:10.8632
[53]	validation_0-rmse:9.83896	validation_1-rmse:10.8626
[54]	validation_0-rmse:9.83109	validation_1-rmse:10.8577
[55]	validation_0-rmse:9.8192	validation_1-rmse:10.8544
[56]	validation_0-rmse:9.81167	validation_1-rmse:10.8568
[57]	validation_0-rmse:9.80971	validation_1-rmse:10.8562
[58]	validation_0-rmse:9.79888	validation_1-rmse:10.8508
[59]	validation_0-rmse:9.79018	validation_1-rmse:10.8517
[60]	validation_0-rmse:9.77796	validation_1-rmse:10.8474
[61]	validation_0-rmse:9.76354	validation_1-rmse:10.8473
[62]	validation_0-rmse:9.75712	validation_1-rmse:10.8467
[63]	validation_0-rmse:9.75149	validation_1-rmse:10.8436
[64]	validation_0-rmse:9.7363	validation_1-rmse:10.8368
[65]	validation_0-rmse:9.73159	validation_1-rmse:10.8358
[66]	validation_0-rmse:9.72083	validation_1-rmse:10.8376
[67]	validation_0-rmse:9.71388	validation_1-rmse:10.8328
[68]	validation_0-rmse:9.70062	validation_1-rmse:10.8309
[69]	validation_0-rmse:9.68877	validation_1-rmse:10.8274
[70]	validation_0-rmse:9.67804	validation_1-rmse:10.825
[71]	validation_0-rmse:9.66857	validation_1-rmse:10.8247
[72]	validation_0-rmse:9.66452	validation_1-rmse:10.8234
[73]	validation_0-rmse:9.65914	validation_1-rmse:10.8218
[74]	validation_0-rmse:9.64795	validation_1-rmse:10.8256
[75]	validation_0-rmse:9.63334	validation_1-rmse:10.8279
[76]	validation_0-rmse:9.62704	validation_1-rmse:10.8268
[77]	validation_0-rmse:9.62068	validation_1-rmse:10.8249
[78]	validation_0-rmse:9.61134	validation_1-rmse:10.8231
[79]	validation_0-rmse:9.60479	validation_1-rmse:10.8207
[80]	validation_0-rmse:9.59429	validation_1-rmse:10.8154
[81]	validation_0-rmse:9.58246	validation_1-rmse:10.8168
[82]	validation_0-rmse:9.57387	validation_1-rmse:10.8167
[83]	validation_0-rmse:9.56855	validation_1-rmse:10.8164
[84]	validation_0-rmse:9.56716	validation_1-rmse:10.8155
[85]	validation_0-rmse:9.55684	validation_1-rmse:10.815
[86]	validation_0-rmse:9.55446	validation_1-rmse:10.8145
[87]	validation_0-rmse:9.54978	validation_1-rmse:10.8149
[88]	validation_0-rmse:9.54047	validation_1-rmse:10.8119
[89]	validation_0-rmse:9.53436	validation_1-rmse:10.811
[90]	validation_0-rmse:9.52264	validation_1-rmse:10.8114
[91]	validation_0-rmse:9.51665	validation_1-rmse:10.8117
[92]	validation_0-rmse:9.51266	validation_1-rmse:10.8102
[93]	validation_0-rmse:9.50319	validation_1-rmse:10.8135
[94]	validation_0-rmse:9.49798	validation_1-rmse:10.8121
[95]	validation_0-rmse:9.4927	validation_1-rmse:10.8115
[96]	validation_0-rmse:9.48973	validation_1-rmse:10.8105
[97]	validation_0-rmse:9.48597	validation_1-rmse:10.8089
[98]	validation_0-rmse:9.48236	validation_1-rmse:10.8088
[99]	validation_0-rmse:9.48213	validation_1-rmse:10.8088
[100]	validation_0-rmse:9.47621	validation_1-rmse:10.8101
[101]	validation_0-rmse:9.47119	validation_1-rmse:10.8092
[102]	validation_0-rmse:9.46442	validation_1-rmse:10.8142
[103]	validation_0-rmse:9.45862	validation_1-rmse:10.8122
[104]	validation_0-rmse:9.45009	validation_1-rmse:10.8125
[105]	validation_0-rmse:9.44749	validation_1-rmse:10.8142
[106]	validation_0-rmse:9.44583	validation_1-rmse:10.8145
[107]	validation_0-rmse:9.44098	validation_1-rmse:10.8151
[108]	validation_0-rmse:9.43302	validation_1-rmse:10.8139
[109]	validation_0-rmse:9.43219	validation_1-rmse:10.8138
[110]	validation_0-rmse:9.42821	validation_1-rmse:10.812
[111]	validation_0-rmse:9.42283	validation_1-rmse:10.8132
[112]	validation_0-rmse:9.42224	validation_1-rmse:10.8132
[113]	validation_0-rmse:9.41739	validation_1-rmse:10.8106
[114]	validation_0-rmse:9.41063	validation_1-rmse:10.8115
[115]	validation_0-rmse:9.40678	validation_1-rmse:10.8118
[116]	validation_0-rmse:9.40117	validation_1-rmse:10.8102
[117]	validation_0-rmse:9.39553	validation_1-rmse:10.8106
[118]	validation_0-rmse:9.38679	validation_1-rmse:10.811
[119]	validation_0-rmse:9.38662	validation_1-rmse:10.811
[120]	validation_0-rmse:9.38034	validation_1-rmse:10.8094
[121]	validation_0-rmse:9.37276	validation_1-rmse:10.8055
[122]	validation_0-rmse:9.36894	validation_1-rmse:10.806
[123]	validation_0-rmse:9.36399	validation_1-rmse:10.8088
[124]	validation_0-rmse:9.36389	validation_1-rmse:10.8083
[125]	validation_0-rmse:9.3638	validation_1-rmse:10.8083
[126]	validation_0-rmse:9.36154	validation_1-rmse:10.809
[127]	validation_0-rmse:9.35938	validation_1-rmse:10.8104
[128]	validation_0-rmse:9.35485	validation_1-rmse:10.8171
[129]	validation_0-rmse:9.35245	validation_1-rmse:10.8185
[130]	validation_0-rmse:9.34775	validation_1-rmse:10.8179
[131]	validation_0-rmse:9.34373	validation_1-rmse:10.8172
[132]	validation_0-rmse:9.33867	validation_1-rmse:10.8181
[133]	validation_0-rmse:9.33583	validation_1-rmse:10.8189
[134]	validation_0-rmse:9.33477	validation_1-rmse:10.8189
[135]	validation_0-rmse:9.32762	validation_1-rmse:10.8165
[136]	validation_0-rmse:9.32481	validation_1-rmse:10.8159
[137]	validation_0-rmse:9.32476	validation_1-rmse:10.8159
[138]	validation_0-rmse:9.32473	validation_1-rmse:10.8159
[139]	validation_0-rmse:9.32348	validation_1-rmse:10.8159
[140]	validation_0-rmse:9.32258	validation_1-rmse:10.8159
[141]	validation_0-rmse:9.32294	validation_1-rmse:10.8162
[142]	validation_0-rmse:9.32149	validation_1-rmse:10.8165
[143]	validation_0-rmse:9.32145	validation_1-rmse:10.8166
[144]	validation_0-rmse:9.32142	validation_1-rmse:10.8166
[145]	validation_0-rmse:9.31889	validation_1-rmse:10.8202
[146]	validation_0-rmse:9.31886	validation_1-rmse:10.8202
[147]	validation_0-rmse:9.31884	validation_1-rmse:10.8202
[148]	validation_0-rmse:9.31881	validation_1-rmse:10.8202
[149]	validation_0-rmse:9.31934	validation_1-rmse:10.8189
[150]	validation_0-rmse:9.31815	validation_1-rmse:10.8209
[151]	validation_0-rmse:9.31604	validation_1-rmse:10.8205
[152]	validation_0-rmse:9.31244	validation_1-rmse:10.8193
[153]	validation_0-rmse:9.30735	validation_1-rmse:10.818
[154]	validation_0-rmse:9.30586	validation_1-rmse:10.818
[155]	validation_0-rmse:9.30562	validation_1-rmse:10.818
[156]	validation_0-rmse:9.3056	validation_1-rmse:10.818
[157]	validation_0-rmse:9.30179	validation_1-rmse:10.8245
[158]	validation_0-rmse:9.30177	validation_1-rmse:10.8245
[159]	validation_0-rmse:9.3001	validation_1-rmse:10.8244
[160]	validation_0-rmse:9.29691	validation_1-rmse:10.8321
[161]	validation_0-rmse:9.29623	validation_1-rmse:10.8313
[162]	validation_0-rmse:9.29471	validation_1-rmse:10.8293
[163]	validation_0-rmse:9.29357	validation_1-rmse:10.8293
[164]	validation_0-rmse:9.29286	validation_1-rmse:10.8292
[165]	validation_0-rmse:9.29285	validation_1-rmse:10.8292
[166]	validation_0-rmse:9.28892	validation_1-rmse:10.8299
[167]	validation_0-rmse:9.28803	validation_1-rmse:10.8299
[168]	validation_0-rmse:9.28802	validation_1-rmse:10.8299
[169]	validation_0-rmse:9.2846	validation_1-rmse:10.83
[170]	validation_0-rmse:9.28164	validation_1-rmse:10.8309
[171]	validation_0-rmse:9.27009	validation_1-rmse:10.8352
Stopping. Best iteration:
[121]	validation_0-rmse:9.37276	validation_1-rmse:10.8055

[0]	validation_0-rmse:24.2159	validation_1-rmse:24.5305
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.2934	validation_1-rmse:22.6178
[2]	validation_0-rmse:20.5999	validation_1-rmse:20.9372
[3]	validation_0-rmse:19.1178	validation_1-rmse:19.4692
[4]	validation_0-rmse:17.8212	validation_1-rmse:18.1873
[5]	validation_0-rmse:16.6971	validation_1-rmse:17.0782
[6]	validation_0-rmse:15.7239	validation_1-rmse:16.126
[7]	validation_0-rmse:14.8838	validation_1-rmse:15.3055
[8]	validation_0-rmse:14.1675	validation_1-rmse:14.6069
[9]	validation_0-rmse:13.5591	validation_1-rmse:14.0127
[10]	validation_0-rmse:13.0361	validation_1-rmse:13.5071
[11]	validation_0-rmse:12.6	validation_1-rmse:13.0872
[12]	validation_0-rmse:12.2337	validation_1-rmse:12.7376
[13]	validation_0-rmse:11.9171	validation_1-rmse:12.4453
[14]	validation_0-rmse:11.6592	validation_1-rmse:12.1956
[15]	validation_0-rmse:11.4379	validation_1-rmse:11.9959
[16]	validation_0-rmse:11.2598	validation_1-rmse:11.8325
[17]	validation_0-rmse:11.1129	validation_1-rmse:11.696
[18]	validation_0-rmse:10.9856	validation_1-rmse:11.582
[19]	validation_0-rmse:10.885	validation_1-rmse:11.488
[20]	validation_0-rmse:10.7853	validation_1-rmse:11.4101
[21]	validation_0-rmse:10.7157	validation_1-rmse:11.3484
[22]	validation_0-rmse:10.6515	validation_1-rmse:11.2936
[23]	validation_0-rmse:10.6007	validation_1-rmse:11.2503
[24]	validation_0-rmse:10.5508	validation_1-rmse:11.2081
[25]	validation_0-rmse:10.5057	validation_1-rmse:11.1761
[26]	validation_0-rmse:10.4679	validation_1-rmse:11.1469
[27]	validation_0-rmse:10.4368	validation_1-rmse:11.1241
[28]	validation_0-rmse:10.4027	validation_1-rmse:11.1026
[29]	validation_0-rmse:10.3793	validation_1-rmse:11.0821
[30]	validation_0-rmse:10.354	validation_1-rmse:11.07
[31]	validation_0-rmse:10.3377	validation_1-rmse:11.0677
[32]	validation_0-rmse:10.323	validation_1-rmse:11.057
[33]	validation_0-rmse:10.3041	validation_1-rmse:11.0458
[34]	validation_0-rmse:10.2874	validation_1-rmse:11.0384
[35]	validation_0-rmse:10.2687	validation_1-rmse:11.0383
[36]	validation_0-rmse:10.2543	validation_1-rmse:11.0298
[37]	validation_0-rmse:10.2431	validation_1-rmse:11.0212
[38]	validation_0-rmse:10.2358	validation_1-rmse:11.0152
[39]	validation_0-rmse:10.2216	validation_1-rmse:11.0075
[40]	validation_0-rmse:10.2064	validation_1-rmse:11.0055
[41]	validation_0-rmse:10.1985	validation_1-rmse:11.0017
[42]	validation_0-rmse:10.1917	validation_1-rmse:10.9995
[43]	validation_0-rmse:10.1826	validation_1-rmse:10.9991
[44]	validation_0-rmse:10.1696	validation_1-rmse:10.9977
[45]	validation_0-rmse:10.1576	validation_1-rmse:10.9966
[46]	validation_0-rmse:10.1411	validation_1-rmse:10.9931
[47]	validation_0-rmse:10.1324	validation_1-rmse:10.9862
[48]	validation_0-rmse:10.1189	validation_1-rmse:10.9826
[49]	validation_0-rmse:10.1144	validation_1-rmse:10.9803
[50]	validation_0-rmse:10.109	validation_1-rmse:10.9828
[51]	validation_0-rmse:10.1004	validation_1-rmse:10.9937
[52]	validation_0-rmse:10.0958	validation_1-rmse:10.9881
[53]	validation_0-rmse:10.0927	validation_1-rmse:10.9855
[54]	validation_0-rmse:10.0804	validation_1-rmse:10.9866
[55]	validation_0-rmse:10.0697	validation_1-rmse:10.9914
[56]	validation_0-rmse:10.0678	validation_1-rmse:10.9924
[57]	validation_0-rmse:10.0613	validation_1-rmse:10.9915
[58]	validation_0-rmse:10.0559	validation_1-rmse:10.99
[59]	validation_0-rmse:10.0524	validation_1-rmse:10.9854
[60]	validation_0-rmse:10.0478	validation_1-rmse:10.9841
[61]	validation_0-rmse:10.0381	validation_1-rmse:10.9855
[62]	validation_0-rmse:10.036	validation_1-rmse:10.9831
[63]	validation_0-rmse:10.0257	validation_1-rmse:10.9862
[64]	validation_0-rmse:10.0212	validation_1-rmse:10.9846
[65]	validation_0-rmse:10.0096	validation_1-rmse:10.9888
[66]	validation_0-rmse:10.0075	validation_1-rmse:10.9891
[67]	validation_0-rmse:9.99939	validation_1-rmse:10.9836
[68]	validation_0-rmse:9.99297	validation_1-rmse:10.9809
[69]	validation_0-rmse:9.99025	validation_1-rmse:10.981
[70]	validation_0-rmse:9.98353	validation_1-rmse:10.9812
[71]	validation_0-rmse:9.97757	validation_1-rmse:10.9901
[72]	validation_0-rmse:9.97209	validation_1-rmse:10.9989
[73]	validation_0-rmse:9.96778	validation_1-rmse:10.9995
[74]	validation_0-rmse:9.96537	validation_1-rmse:10.9997
[75]	validation_0-rmse:9.96022	validation_1-rmse:11.0032
[76]	validation_0-rmse:9.95652	validation_1-rmse:11.0009
[77]	validation_0-rmse:9.94945	validation_1-rmse:10.9996
[78]	validation_0-rmse:9.94127	validation_1-rmse:11.002
[79]	validation_0-rmse:9.93702	validation_1-rmse:10.9986
[80]	validation_0-rmse:9.93278	validation_1-rmse:10.9979
[81]	validation_0-rmse:9.92616	validation_1-rmse:10.9949
[82]	validation_0-rmse:9.91941	validation_1-rmse:10.991
[83]	validation_0-rmse:9.91431	validation_1-rmse:10.9913
[84]	validation_0-rmse:9.90831	validation_1-rmse:10.9902
[85]	validation_0-rmse:9.90261	validation_1-rmse:10.9932
[86]	validation_0-rmse:9.89771	validation_1-rmse:10.9958
[87]	validation_0-rmse:9.89552	validation_1-rmse:10.9979
[88]	validation_0-rmse:9.88916	validation_1-rmse:10.9928
[89]	validation_0-rmse:9.8877	validation_1-rmse:10.9896
[90]	validation_0-rmse:9.88406	validation_1-rmse:10.9995
[91]	validation_0-rmse:9.87565	validation_1-rmse:10.9948
[92]	validation_0-rmse:9.87299	validation_1-rmse:10.9937
[93]	validation_0-rmse:9.86659	validation_1-rmse:10.9943
[94]	validation_0-rmse:9.8616	validation_1-rmse:10.9948
[95]	validation_0-rmse:9.8579	validation_1-rmse:10.9979
[96]	validation_0-rmse:9.8563	validation_1-rmse:10.9987
[97]	validation_0-rmse:9.85502	validation_1-rmse:11.0053
[98]	validation_0-rmse:9.85176	validation_1-rmse:11.0074
[99]	validation_0-rmse:9.84826	validation_1-rmse:11.01
Stopping. Best iteration:
[49]	validation_0-rmse:10.1144	validation_1-rmse:10.9803

[0]	validation_0-rmse:24.231	validation_1-rmse:24.5477
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.314	validation_1-rmse:22.6404
[2]	validation_0-rmse:20.6231	validation_1-rmse:20.9632
[3]	validation_0-rmse:19.1439	validation_1-rmse:19.505
[4]	validation_0-rmse:17.9513	validation_1-rmse:18.3314
[5]	validation_0-rmse:16.9099	validation_1-rmse:17.3081
[6]	validation_0-rmse:15.9088	validation_1-rmse:16.3254
[7]	validation_0-rmse:15.1346	validation_1-rmse:15.5688
[8]	validation_0-rmse:14.3935	validation_1-rmse:14.8454
[9]	validation_0-rmse:13.7546	validation_1-rmse:14.2247
[10]	validation_0-rmse:13.1952	validation_1-rmse:13.6905
[11]	validation_0-rmse:12.7253	validation_1-rmse:13.2433
[12]	validation_0-rmse:12.3209	validation_1-rmse:12.8635
[13]	validation_0-rmse:12.0244	validation_1-rmse:12.585
[14]	validation_0-rmse:11.7611	validation_1-rmse:12.342
[15]	validation_0-rmse:11.5161	validation_1-rmse:12.1149
[16]	validation_0-rmse:11.3208	validation_1-rmse:11.9395
[17]	validation_0-rmse:11.1441	validation_1-rmse:11.7772
[18]	validation_0-rmse:11.0067	validation_1-rmse:11.6561
[19]	validation_0-rmse:10.8772	validation_1-rmse:11.5378
[20]	validation_0-rmse:10.7899	validation_1-rmse:11.4605
[21]	validation_0-rmse:10.6881	validation_1-rmse:11.3785
[22]	validation_0-rmse:10.6095	validation_1-rmse:11.3089
[23]	validation_0-rmse:10.5325	validation_1-rmse:11.2508
[24]	validation_0-rmse:10.4611	validation_1-rmse:11.1982
[25]	validation_0-rmse:10.4018	validation_1-rmse:11.1575
[26]	validation_0-rmse:10.3501	validation_1-rmse:11.1176
[27]	validation_0-rmse:10.3062	validation_1-rmse:11.091
[28]	validation_0-rmse:10.2627	validation_1-rmse:11.064
[29]	validation_0-rmse:10.2274	validation_1-rmse:11.0389
[30]	validation_0-rmse:10.1968	validation_1-rmse:11.02
[31]	validation_0-rmse:10.1704	validation_1-rmse:11.009
[32]	validation_0-rmse:10.1522	validation_1-rmse:10.9973
[33]	validation_0-rmse:10.1284	validation_1-rmse:10.9876
[34]	validation_0-rmse:10.1081	validation_1-rmse:10.9738
[35]	validation_0-rmse:10.0955	validation_1-rmse:10.9679
[36]	validation_0-rmse:10.0773	validation_1-rmse:10.9581
[37]	validation_0-rmse:10.0639	validation_1-rmse:10.9499
[38]	validation_0-rmse:10.049	validation_1-rmse:10.9493
[39]	validation_0-rmse:10.0373	validation_1-rmse:10.9456
[40]	validation_0-rmse:10.0185	validation_1-rmse:10.9413
[41]	validation_0-rmse:10.0078	validation_1-rmse:10.9393
[42]	validation_0-rmse:9.99796	validation_1-rmse:10.9347
[43]	validation_0-rmse:9.98474	validation_1-rmse:10.9289
[44]	validation_0-rmse:9.97565	validation_1-rmse:10.9239
[45]	validation_0-rmse:9.97121	validation_1-rmse:10.9241
[46]	validation_0-rmse:9.95761	validation_1-rmse:10.9182
[47]	validation_0-rmse:9.94004	validation_1-rmse:10.9119
[48]	validation_0-rmse:9.92803	validation_1-rmse:10.9118
[49]	validation_0-rmse:9.92019	validation_1-rmse:10.9072
[50]	validation_0-rmse:9.91398	validation_1-rmse:10.9106
[51]	validation_0-rmse:9.90076	validation_1-rmse:10.9041
[52]	validation_0-rmse:9.89362	validation_1-rmse:10.9026
[53]	validation_0-rmse:9.88867	validation_1-rmse:10.9005
[54]	validation_0-rmse:9.87518	validation_1-rmse:10.8975
[55]	validation_0-rmse:9.85969	validation_1-rmse:10.8915
[56]	validation_0-rmse:9.85692	validation_1-rmse:10.8932
[57]	validation_0-rmse:9.84816	validation_1-rmse:10.8912
[58]	validation_0-rmse:9.84316	validation_1-rmse:10.8891
[59]	validation_0-rmse:9.83748	validation_1-rmse:10.8853
[60]	validation_0-rmse:9.83409	validation_1-rmse:10.8858
[61]	validation_0-rmse:9.82493	validation_1-rmse:10.8872
[62]	validation_0-rmse:9.8212	validation_1-rmse:10.887
[63]	validation_0-rmse:9.81987	validation_1-rmse:10.8879
[64]	validation_0-rmse:9.81519	validation_1-rmse:10.8868
[65]	validation_0-rmse:9.80957	validation_1-rmse:10.8852
[66]	validation_0-rmse:9.80696	validation_1-rmse:10.8877
[67]	validation_0-rmse:9.79951	validation_1-rmse:10.8837
[68]	validation_0-rmse:9.78738	validation_1-rmse:10.8768
[69]	validation_0-rmse:9.78531	validation_1-rmse:10.8782
[70]	validation_0-rmse:9.77652	validation_1-rmse:10.8752
[71]	validation_0-rmse:9.76896	validation_1-rmse:10.8764
[72]	validation_0-rmse:9.75915	validation_1-rmse:10.8824
[73]	validation_0-rmse:9.74836	validation_1-rmse:10.8859
[74]	validation_0-rmse:9.74549	validation_1-rmse:10.886
[75]	validation_0-rmse:9.74171	validation_1-rmse:10.8876
[76]	validation_0-rmse:9.7356	validation_1-rmse:10.8877
[77]	validation_0-rmse:9.7261	validation_1-rmse:10.8908
[78]	validation_0-rmse:9.71907	validation_1-rmse:10.8898
[79]	validation_0-rmse:9.71884	validation_1-rmse:10.8898
[80]	validation_0-rmse:9.7124	validation_1-rmse:10.8876
[81]	validation_0-rmse:9.70052	validation_1-rmse:10.882
[82]	validation_0-rmse:9.69722	validation_1-rmse:10.8819
[83]	validation_0-rmse:9.69233	validation_1-rmse:10.8791
[84]	validation_0-rmse:9.68987	validation_1-rmse:10.8816
[85]	validation_0-rmse:9.68205	validation_1-rmse:10.8781
[86]	validation_0-rmse:9.67368	validation_1-rmse:10.8748
[87]	validation_0-rmse:9.67192	validation_1-rmse:10.8744
[88]	validation_0-rmse:9.6624	validation_1-rmse:10.8682
[89]	validation_0-rmse:9.65918	validation_1-rmse:10.867
[90]	validation_0-rmse:9.64611	validation_1-rmse:10.8602
[91]	validation_0-rmse:9.63749	validation_1-rmse:10.8595
[92]	validation_0-rmse:9.63379	validation_1-rmse:10.8568
[93]	validation_0-rmse:9.62242	validation_1-rmse:10.8564
[94]	validation_0-rmse:9.6205	validation_1-rmse:10.8563
[95]	validation_0-rmse:9.62011	validation_1-rmse:10.8562
[96]	validation_0-rmse:9.61789	validation_1-rmse:10.8565
[97]	validation_0-rmse:9.61619	validation_1-rmse:10.8562
[98]	validation_0-rmse:9.60709	validation_1-rmse:10.8537
[99]	validation_0-rmse:9.60694	validation_1-rmse:10.8536
[100]	validation_0-rmse:9.59993	validation_1-rmse:10.853
[101]	validation_0-rmse:9.59139	validation_1-rmse:10.8538
[102]	validation_0-rmse:9.59128	validation_1-rmse:10.8538
[103]	validation_0-rmse:9.5806	validation_1-rmse:10.8493
[104]	validation_0-rmse:9.57212	validation_1-rmse:10.8466
[105]	validation_0-rmse:9.56992	validation_1-rmse:10.8466
[106]	validation_0-rmse:9.56981	validation_1-rmse:10.8465
[107]	validation_0-rmse:9.5602	validation_1-rmse:10.844
[108]	validation_0-rmse:9.55534	validation_1-rmse:10.8423
[109]	validation_0-rmse:9.55226	validation_1-rmse:10.8441
[110]	validation_0-rmse:9.54932	validation_1-rmse:10.8426
[111]	validation_0-rmse:9.54894	validation_1-rmse:10.8441
[112]	validation_0-rmse:9.54378	validation_1-rmse:10.8429
[113]	validation_0-rmse:9.52884	validation_1-rmse:10.8371
[114]	validation_0-rmse:9.51969	validation_1-rmse:10.8376
[115]	validation_0-rmse:9.51736	validation_1-rmse:10.8387
[116]	validation_0-rmse:9.51301	validation_1-rmse:10.8368
[117]	validation_0-rmse:9.50654	validation_1-rmse:10.8367
[118]	validation_0-rmse:9.50283	validation_1-rmse:10.8397
[119]	validation_0-rmse:9.50127	validation_1-rmse:10.8396
[120]	validation_0-rmse:9.49975	validation_1-rmse:10.8406
[121]	validation_0-rmse:9.49776	validation_1-rmse:10.8409
[122]	validation_0-rmse:9.49515	validation_1-rmse:10.8406
[123]	validation_0-rmse:9.49152	validation_1-rmse:10.84
[124]	validation_0-rmse:9.48401	validation_1-rmse:10.8391
[125]	validation_0-rmse:9.48266	validation_1-rmse:10.8388
[126]	validation_0-rmse:9.482	validation_1-rmse:10.8398
[127]	validation_0-rmse:9.48167	validation_1-rmse:10.8389
[128]	validation_0-rmse:9.47233	validation_1-rmse:10.8361
[129]	validation_0-rmse:9.46771	validation_1-rmse:10.8338
[130]	validation_0-rmse:9.46599	validation_1-rmse:10.8344
[131]	validation_0-rmse:9.46148	validation_1-rmse:10.8375
[132]	validation_0-rmse:9.46107	validation_1-rmse:10.8384
[133]	validation_0-rmse:9.45375	validation_1-rmse:10.8389
[134]	validation_0-rmse:9.45203	validation_1-rmse:10.8406
[135]	validation_0-rmse:9.44909	validation_1-rmse:10.84
[136]	validation_0-rmse:9.4439	validation_1-rmse:10.8388
[137]	validation_0-rmse:9.43999	validation_1-rmse:10.836
[138]	validation_0-rmse:9.43115	validation_1-rmse:10.838
[139]	validation_0-rmse:9.42883	validation_1-rmse:10.838
[140]	validation_0-rmse:9.42723	validation_1-rmse:10.8373
[141]	validation_0-rmse:9.42484	validation_1-rmse:10.8373
[142]	validation_0-rmse:9.42106	validation_1-rmse:10.84
[143]	validation_0-rmse:9.41943	validation_1-rmse:10.8393
[144]	validation_0-rmse:9.41508	validation_1-rmse:10.8424
[145]	validation_0-rmse:9.40988	validation_1-rmse:10.8455
[146]	validation_0-rmse:9.40537	validation_1-rmse:10.8443
[147]	validation_0-rmse:9.40291	validation_1-rmse:10.8447
[148]	validation_0-rmse:9.40286	validation_1-rmse:10.8445
[149]	validation_0-rmse:9.40074	validation_1-rmse:10.8426
[150]	validation_0-rmse:9.39663	validation_1-rmse:10.8443
[151]	validation_0-rmse:9.3938	validation_1-rmse:10.8447
[152]	validation_0-rmse:9.38931	validation_1-rmse:10.8434
[153]	validation_0-rmse:9.38801	validation_1-rmse:10.8441
[154]	validation_0-rmse:9.38508	validation_1-rmse:10.846
[155]	validation_0-rmse:9.38082	validation_1-rmse:10.8466
[156]	validation_0-rmse:9.37865	validation_1-rmse:10.8463
[157]	validation_0-rmse:9.37861	validation_1-rmse:10.8463
[158]	validation_0-rmse:9.37468	validation_1-rmse:10.8435
[159]	validation_0-rmse:9.3742	validation_1-rmse:10.8439
[160]	validation_0-rmse:9.36682	validation_1-rmse:10.8397
[161]	validation_0-rmse:9.35749	validation_1-rmse:10.8368
[162]	validation_0-rmse:9.35742	validation_1-rmse:10.8359
[163]	validation_0-rmse:9.35627	validation_1-rmse:10.8351
[164]	validation_0-rmse:9.35168	validation_1-rmse:10.837
[165]	validation_0-rmse:9.3467	validation_1-rmse:10.8439
[166]	validation_0-rmse:9.34657	validation_1-rmse:10.8449
[167]	validation_0-rmse:9.34564	validation_1-rmse:10.8451
[168]	validation_0-rmse:9.3456	validation_1-rmse:10.8451
[169]	validation_0-rmse:9.34446	validation_1-rmse:10.8448
[170]	validation_0-rmse:9.34287	validation_1-rmse:10.8437
[171]	validation_0-rmse:9.33882	validation_1-rmse:10.8419
[172]	validation_0-rmse:9.33828	validation_1-rmse:10.8418
[173]	validation_0-rmse:9.32966	validation_1-rmse:10.8477
[174]	validation_0-rmse:9.32278	validation_1-rmse:10.8457
[175]	validation_0-rmse:9.31807	validation_1-rmse:10.8437
[176]	validation_0-rmse:9.31065	validation_1-rmse:10.841
[177]	validation_0-rmse:9.30294	validation_1-rmse:10.8375
[178]	validation_0-rmse:9.29857	validation_1-rmse:10.8407
[179]	validation_0-rmse:9.29811	validation_1-rmse:10.8408
Stopping. Best iteration:
[129]	validation_0-rmse:9.46771	validation_1-rmse:10.8338

[0]	validation_0-rmse:24.2241	validation_1-rmse:24.5395
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3085	validation_1-rmse:22.6332
[2]	validation_0-rmse:20.6229	validation_1-rmse:20.9576
[3]	validation_0-rmse:19.1479	validation_1-rmse:19.4932
[4]	validation_0-rmse:17.8574	validation_1-rmse:18.2186
[5]	validation_0-rmse:16.7385	validation_1-rmse:17.1095
[6]	validation_0-rmse:15.7689	validation_1-rmse:16.1555
[7]	validation_0-rmse:14.9351	validation_1-rmse:15.3377
[8]	validation_0-rmse:14.2206	validation_1-rmse:14.6365
[9]	validation_0-rmse:13.6116	validation_1-rmse:14.0378
[10]	validation_0-rmse:13.0931	validation_1-rmse:13.5331
[11]	validation_0-rmse:12.6537	validation_1-rmse:13.1114
[12]	validation_0-rmse:12.2831	validation_1-rmse:12.7543
[13]	validation_0-rmse:11.9722	validation_1-rmse:12.4573
[14]	validation_0-rmse:11.7151	validation_1-rmse:12.2104
[15]	validation_0-rmse:11.5021	validation_1-rmse:12.0064
[16]	validation_0-rmse:11.3196	validation_1-rmse:11.8354
[17]	validation_0-rmse:11.1711	validation_1-rmse:11.6937
[18]	validation_0-rmse:11.0452	validation_1-rmse:11.5749
[19]	validation_0-rmse:10.9422	validation_1-rmse:11.4755
[20]	validation_0-rmse:10.848	validation_1-rmse:11.3924
[21]	validation_0-rmse:10.7761	validation_1-rmse:11.3259
[22]	validation_0-rmse:10.7058	validation_1-rmse:11.2647
[23]	validation_0-rmse:10.6478	validation_1-rmse:11.2198
[24]	validation_0-rmse:10.6031	validation_1-rmse:11.1784
[25]	validation_0-rmse:10.5539	validation_1-rmse:11.1428
[26]	validation_0-rmse:10.5183	validation_1-rmse:11.1097
[27]	validation_0-rmse:10.4886	validation_1-rmse:11.0866
[28]	validation_0-rmse:10.459	validation_1-rmse:11.0668
[29]	validation_0-rmse:10.4353	validation_1-rmse:11.0478
[30]	validation_0-rmse:10.4107	validation_1-rmse:11.0337
[31]	validation_0-rmse:10.3865	validation_1-rmse:11.0241
[32]	validation_0-rmse:10.3663	validation_1-rmse:11.016
[33]	validation_0-rmse:10.3547	validation_1-rmse:11.0085
[34]	validation_0-rmse:10.3349	validation_1-rmse:11.001
[35]	validation_0-rmse:10.3159	validation_1-rmse:10.9974
[36]	validation_0-rmse:10.303	validation_1-rmse:10.99
[37]	validation_0-rmse:10.2928	validation_1-rmse:10.9825
[38]	validation_0-rmse:10.2776	validation_1-rmse:10.9802
[39]	validation_0-rmse:10.2706	validation_1-rmse:10.9753
[40]	validation_0-rmse:10.2583	validation_1-rmse:10.9705
[41]	validation_0-rmse:10.2457	validation_1-rmse:10.9743
[42]	validation_0-rmse:10.2268	validation_1-rmse:10.9718
[43]	validation_0-rmse:10.2202	validation_1-rmse:10.9664
[44]	validation_0-rmse:10.2065	validation_1-rmse:10.9674
[45]	validation_0-rmse:10.1932	validation_1-rmse:10.9666
[46]	validation_0-rmse:10.186	validation_1-rmse:10.9649
[47]	validation_0-rmse:10.1711	validation_1-rmse:10.9603
[48]	validation_0-rmse:10.1533	validation_1-rmse:10.9537
[49]	validation_0-rmse:10.1435	validation_1-rmse:10.9497
[50]	validation_0-rmse:10.1336	validation_1-rmse:10.9514
[51]	validation_0-rmse:10.1272	validation_1-rmse:10.9474
[52]	validation_0-rmse:10.1235	validation_1-rmse:10.9448
[53]	validation_0-rmse:10.121	validation_1-rmse:10.9428
[54]	validation_0-rmse:10.1105	validation_1-rmse:10.9379
[55]	validation_0-rmse:10.1013	validation_1-rmse:10.9366
[56]	validation_0-rmse:10.0984	validation_1-rmse:10.937
[57]	validation_0-rmse:10.0945	validation_1-rmse:10.937
[58]	validation_0-rmse:10.0892	validation_1-rmse:10.9352
[59]	validation_0-rmse:10.0845	validation_1-rmse:10.934
[60]	validation_0-rmse:10.077	validation_1-rmse:10.9344
[61]	validation_0-rmse:10.0649	validation_1-rmse:10.9357
[62]	validation_0-rmse:10.0604	validation_1-rmse:10.9339
[63]	validation_0-rmse:10.0564	validation_1-rmse:10.9322
[64]	validation_0-rmse:10.0531	validation_1-rmse:10.931
[65]	validation_0-rmse:10.0443	validation_1-rmse:10.9353
[66]	validation_0-rmse:10.0392	validation_1-rmse:10.935
[67]	validation_0-rmse:10.0329	validation_1-rmse:10.9329
[68]	validation_0-rmse:10.0261	validation_1-rmse:10.9291
[69]	validation_0-rmse:10.0195	validation_1-rmse:10.9315
[70]	validation_0-rmse:10.0151	validation_1-rmse:10.9308
[71]	validation_0-rmse:10.0096	validation_1-rmse:10.9351
[72]	validation_0-rmse:10.0021	validation_1-rmse:10.9406
[73]	validation_0-rmse:9.99596	validation_1-rmse:10.9446
[74]	validation_0-rmse:9.9902	validation_1-rmse:10.9497
[75]	validation_0-rmse:9.98492	validation_1-rmse:10.9511
[76]	validation_0-rmse:9.97782	validation_1-rmse:10.9547
[77]	validation_0-rmse:9.97032	validation_1-rmse:10.9579
[78]	validation_0-rmse:9.96876	validation_1-rmse:10.9578
[79]	validation_0-rmse:9.96673	validation_1-rmse:10.9585
[80]	validation_0-rmse:9.96562	validation_1-rmse:10.9596
[81]	validation_0-rmse:9.95703	validation_1-rmse:10.9629
[82]	validation_0-rmse:9.95075	validation_1-rmse:10.9618
[83]	validation_0-rmse:9.94477	validation_1-rmse:10.9575
[84]	validation_0-rmse:9.94358	validation_1-rmse:10.9557
[85]	validation_0-rmse:9.93772	validation_1-rmse:10.9631
[86]	validation_0-rmse:9.93233	validation_1-rmse:10.9623
[87]	validation_0-rmse:9.92598	validation_1-rmse:10.9607
[88]	validation_0-rmse:9.91902	validation_1-rmse:10.9565
[89]	validation_0-rmse:9.9127	validation_1-rmse:10.9578
[90]	validation_0-rmse:9.90801	validation_1-rmse:10.965
[91]	validation_0-rmse:9.904	validation_1-rmse:10.9627
[92]	validation_0-rmse:9.89821	validation_1-rmse:10.9627
[93]	validation_0-rmse:9.88899	validation_1-rmse:10.9656
[94]	validation_0-rmse:9.88438	validation_1-rmse:10.9648
[95]	validation_0-rmse:9.88149	validation_1-rmse:10.9614
[96]	validation_0-rmse:9.87925	validation_1-rmse:10.9612
[97]	validation_0-rmse:9.87706	validation_1-rmse:10.9603
[98]	validation_0-rmse:9.87283	validation_1-rmse:10.9582
[99]	validation_0-rmse:9.86605	validation_1-rmse:10.9665
[100]	validation_0-rmse:9.86173	validation_1-rmse:10.9687
[101]	validation_0-rmse:9.85813	validation_1-rmse:10.9719
[102]	validation_0-rmse:9.85271	validation_1-rmse:10.9764
[103]	validation_0-rmse:9.84904	validation_1-rmse:10.9763
[104]	validation_0-rmse:9.84113	validation_1-rmse:10.9766
[105]	validation_0-rmse:9.83804	validation_1-rmse:10.9769
[106]	validation_0-rmse:9.83506	validation_1-rmse:10.978
[107]	validation_0-rmse:9.82978	validation_1-rmse:10.9765
[108]	validation_0-rmse:9.82918	validation_1-rmse:10.9768
[109]	validation_0-rmse:9.82579	validation_1-rmse:10.9808
[110]	validation_0-rmse:9.82403	validation_1-rmse:10.98
[111]	validation_0-rmse:9.81693	validation_1-rmse:10.9797
[112]	validation_0-rmse:9.81206	validation_1-rmse:10.9788
[113]	validation_0-rmse:9.80709	validation_1-rmse:10.9797
[114]	validation_0-rmse:9.79944	validation_1-rmse:10.979
[115]	validation_0-rmse:9.79675	validation_1-rmse:10.9794
[116]	validation_0-rmse:9.79279	validation_1-rmse:10.9791
[117]	validation_0-rmse:9.78451	validation_1-rmse:10.9771
[118]	validation_0-rmse:9.78338	validation_1-rmse:10.9771
Stopping. Best iteration:
[68]	validation_0-rmse:10.0261	validation_1-rmse:10.9291

[0]	validation_0-rmse:24.2245	validation_1-rmse:24.5397
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.3092	validation_1-rmse:22.6332
[2]	validation_0-rmse:20.6237	validation_1-rmse:20.9576
[3]	validation_0-rmse:19.1494	validation_1-rmse:19.4948
[4]	validation_0-rmse:17.8595	validation_1-rmse:18.2198
[5]	validation_0-rmse:16.7408	validation_1-rmse:17.11
[6]	validation_0-rmse:15.7719	validation_1-rmse:16.1567
[7]	validation_0-rmse:14.9383	validation_1-rmse:15.3403
[8]	validation_0-rmse:14.2248	validation_1-rmse:14.6388
[9]	validation_0-rmse:13.6173	validation_1-rmse:14.0412
[10]	validation_0-rmse:13.1007	validation_1-rmse:13.5361
[11]	validation_0-rmse:12.6625	validation_1-rmse:13.1133
[12]	validation_0-rmse:12.2944	validation_1-rmse:12.7565
[13]	validation_0-rmse:11.9828	validation_1-rmse:12.458
[14]	validation_0-rmse:11.7274	validation_1-rmse:12.2136
[15]	validation_0-rmse:11.5123	validation_1-rmse:12.0095
[16]	validation_0-rmse:11.3295	validation_1-rmse:11.8382
[17]	validation_0-rmse:11.1803	validation_1-rmse:11.6983
[18]	validation_0-rmse:11.0521	validation_1-rmse:11.5778
[19]	validation_0-rmse:10.9474	validation_1-rmse:11.4802
[20]	validation_0-rmse:10.8562	validation_1-rmse:11.3975
[21]	validation_0-rmse:10.7822	validation_1-rmse:11.3306
[22]	validation_0-rmse:10.7193	validation_1-rmse:11.2749
[23]	validation_0-rmse:10.6627	validation_1-rmse:11.2301
[24]	validation_0-rmse:10.6146	validation_1-rmse:11.189
[25]	validation_0-rmse:10.5728	validation_1-rmse:11.1549
[26]	validation_0-rmse:10.5347	validation_1-rmse:11.1228
[27]	validation_0-rmse:10.5066	validation_1-rmse:11.098
[28]	validation_0-rmse:10.4763	validation_1-rmse:11.0787
[29]	validation_0-rmse:10.4536	validation_1-rmse:11.0611
[30]	validation_0-rmse:10.433	validation_1-rmse:11.0476
[31]	validation_0-rmse:10.4097	validation_1-rmse:11.038
[32]	validation_0-rmse:10.3868	validation_1-rmse:11.0303
[33]	validation_0-rmse:10.3728	validation_1-rmse:11.0215
[34]	validation_0-rmse:10.3532	validation_1-rmse:11.0136
[35]	validation_0-rmse:10.3372	validation_1-rmse:11.0094
[36]	validation_0-rmse:10.324	validation_1-rmse:11.0009
[37]	validation_0-rmse:10.3175	validation_1-rmse:10.996
[38]	validation_0-rmse:10.3051	validation_1-rmse:10.9924
[39]	validation_0-rmse:10.292	validation_1-rmse:10.9875
[40]	validation_0-rmse:10.2797	validation_1-rmse:10.9866
[41]	validation_0-rmse:10.268	validation_1-rmse:10.9883
[42]	validation_0-rmse:10.2523	validation_1-rmse:10.9827
[43]	validation_0-rmse:10.2443	validation_1-rmse:10.9813
[44]	validation_0-rmse:10.2335	validation_1-rmse:10.9818
[45]	validation_0-rmse:10.2206	validation_1-rmse:10.9807
[46]	validation_0-rmse:10.209	validation_1-rmse:10.9793
[47]	validation_0-rmse:10.2011	validation_1-rmse:10.9767
[48]	validation_0-rmse:10.1943	validation_1-rmse:10.9793
[49]	validation_0-rmse:10.1857	validation_1-rmse:10.9752
[50]	validation_0-rmse:10.1768	validation_1-rmse:10.9723
[51]	validation_0-rmse:10.1618	validation_1-rmse:10.9713
[52]	validation_0-rmse:10.1556	validation_1-rmse:10.9686
[53]	validation_0-rmse:10.1531	validation_1-rmse:10.9662
[54]	validation_0-rmse:10.1404	validation_1-rmse:10.9641
[55]	validation_0-rmse:10.1334	validation_1-rmse:10.9657
[56]	validation_0-rmse:10.1315	validation_1-rmse:10.9665
[57]	validation_0-rmse:10.126	validation_1-rmse:10.9653
[58]	validation_0-rmse:10.1218	validation_1-rmse:10.9644
[59]	validation_0-rmse:10.1197	validation_1-rmse:10.9642
[60]	validation_0-rmse:10.1112	validation_1-rmse:10.9675
[61]	validation_0-rmse:10.0983	validation_1-rmse:10.9691
[62]	validation_0-rmse:10.0919	validation_1-rmse:10.9684
[63]	validation_0-rmse:10.0876	validation_1-rmse:10.9668
[64]	validation_0-rmse:10.0751	validation_1-rmse:10.9579
[65]	validation_0-rmse:10.0695	validation_1-rmse:10.9632
[66]	validation_0-rmse:10.0648	validation_1-rmse:10.9606
[67]	validation_0-rmse:10.053	validation_1-rmse:10.9577
[68]	validation_0-rmse:10.0472	validation_1-rmse:10.9534
[69]	validation_0-rmse:10.0426	validation_1-rmse:10.9547
[70]	validation_0-rmse:10.0372	validation_1-rmse:10.9545
[71]	validation_0-rmse:10.0333	validation_1-rmse:10.9563
[72]	validation_0-rmse:10.0265	validation_1-rmse:10.9573
[73]	validation_0-rmse:10.0195	validation_1-rmse:10.9611
[74]	validation_0-rmse:10.0144	validation_1-rmse:10.9628
[75]	validation_0-rmse:10.0093	validation_1-rmse:10.9644
[76]	validation_0-rmse:10.0041	validation_1-rmse:10.966
[77]	validation_0-rmse:10.0006	validation_1-rmse:10.9681
[78]	validation_0-rmse:9.99797	validation_1-rmse:10.9679
[79]	validation_0-rmse:9.9934	validation_1-rmse:10.9709
[80]	validation_0-rmse:9.98991	validation_1-rmse:10.9694
[81]	validation_0-rmse:9.98087	validation_1-rmse:10.9643
[82]	validation_0-rmse:9.9779	validation_1-rmse:10.9675
[83]	validation_0-rmse:9.97184	validation_1-rmse:10.9629
[84]	validation_0-rmse:9.9666	validation_1-rmse:10.9609
[85]	validation_0-rmse:9.96079	validation_1-rmse:10.9668
[86]	validation_0-rmse:9.95927	validation_1-rmse:10.9672
[87]	validation_0-rmse:9.95336	validation_1-rmse:10.9656
[88]	validation_0-rmse:9.94948	validation_1-rmse:10.9617
[89]	validation_0-rmse:9.94181	validation_1-rmse:10.9617
[90]	validation_0-rmse:9.93806	validation_1-rmse:10.9675
[91]	validation_0-rmse:9.93403	validation_1-rmse:10.966
[92]	validation_0-rmse:9.93203	validation_1-rmse:10.9677
[93]	validation_0-rmse:9.92277	validation_1-rmse:10.9652
[94]	validation_0-rmse:9.91912	validation_1-rmse:10.9662
[95]	validation_0-rmse:9.91634	validation_1-rmse:10.9645
[96]	validation_0-rmse:9.91471	validation_1-rmse:10.9646
[97]	validation_0-rmse:9.91088	validation_1-rmse:10.9631
[98]	validation_0-rmse:9.90484	validation_1-rmse:10.9581
[99]	validation_0-rmse:9.90205	validation_1-rmse:10.9653
[100]	validation_0-rmse:9.89779	validation_1-rmse:10.9681
[101]	validation_0-rmse:9.89152	validation_1-rmse:10.9679
[102]	validation_0-rmse:9.88892	validation_1-rmse:10.9752
[103]	validation_0-rmse:9.88478	validation_1-rmse:10.9738
[104]	validation_0-rmse:9.87544	validation_1-rmse:10.9701
[105]	validation_0-rmse:9.87369	validation_1-rmse:10.9699
[106]	validation_0-rmse:9.87067	validation_1-rmse:10.9686
[107]	validation_0-rmse:9.8652	validation_1-rmse:10.9677
[108]	validation_0-rmse:9.86479	validation_1-rmse:10.9672
[109]	validation_0-rmse:9.86295	validation_1-rmse:10.9704
[110]	validation_0-rmse:9.8601	validation_1-rmse:10.9707
[111]	validation_0-rmse:9.85617	validation_1-rmse:10.9726
[112]	validation_0-rmse:9.84967	validation_1-rmse:10.9727
[113]	validation_0-rmse:9.84339	validation_1-rmse:10.9797
[114]	validation_0-rmse:9.83761	validation_1-rmse:10.9788
[115]	validation_0-rmse:9.83456	validation_1-rmse:10.9799
[116]	validation_0-rmse:9.8312	validation_1-rmse:10.9787
[117]	validation_0-rmse:9.8266	validation_1-rmse:10.9764
[118]	validation_0-rmse:9.81889	validation_1-rmse:10.9734
Stopping. Best iteration:
[68]	validation_0-rmse:10.0472	validation_1-rmse:10.9534

[0]	validation_0-rmse:24.222	validation_1-rmse:24.5437
Multiple eval metrics have been passed: 'validation_1-rmse' will be used for early stopping.

Will train until validation_1-rmse hasn't improved in 50 rounds.
[1]	validation_0-rmse:22.301	validation_1-rmse:22.6319
[2]	validation_0-rmse:20.6044	validation_1-rmse:20.9517
[3]	validation_0-rmse:19.1175	validation_1-rmse:19.4817
[4]	validation_0-rmse:17.8118	validation_1-rmse:18.2075
[5]	validation_0-rmse:16.6792	validation_1-rmse:17.0955
[6]	validation_0-rmse:15.6975	validation_1-rmse:16.1386
[7]	validation_0-rmse:14.8581	validation_1-rmse:15.3172
[8]	validation_0-rmse:14.145	validation_1-rmse:14.6266
[9]	validation_0-rmse:13.5342	validation_1-rmse:14.0392
[10]	validation_0-rmse:12.9979	validation_1-rmse:13.5336
[11]	validation_0-rmse:12.5509	validation_1-rmse:13.1082
[12]	validation_0-rmse:12.1614	validation_1-rmse:12.7541
[13]	validation_0-rmse:11.8227	validation_1-rmse:12.46
[14]	validation_0-rmse:11.543	validation_1-rmse:12.216
[15]	validation_0-rmse:11.3251	validation_1-rmse:12.0158
[16]	validation_0-rmse:11.1211	validation_1-rmse:11.8457
[17]	validation_0-rmse:10.9637	validation_1-rmse:11.7004
[18]	validation_0-rmse:10.8293	validation_1-rmse:11.5776
[19]	validation_0-rmse:10.7235	validation_1-rmse:11.479
[20]	validation_0-rmse:10.6322	validation_1-rmse:11.3982
[21]	validation_0-rmse:10.5405	validation_1-rmse:11.3329
[22]	validation_0-rmse:10.4604	validation_1-rmse:11.2826
[23]	validation_0-rmse:10.3979	validation_1-rmse:11.2361
[24]	validation_0-rmse:10.3292	validation_1-rmse:11.1915
[25]	validation_0-rmse:10.2736	validation_1-rmse:11.1561
[26]	validation_0-rmse:10.2319	validation_1-rmse:11.1254
[27]	validation_0-rmse:10.1931	validation_1-rmse:11.1028
[28]	validation_0-rmse:10.1456	validation_1-rmse:11.0865
[29]	validation_0-rmse:10.1199	validation_1-rmse:11.0671
[30]	validation_0-rmse:10.0878	validation_1-rmse:11.0546
[31]	validation_0-rmse:10.0696	validation_1-rmse:11.0408
[32]	validation_0-rmse:10.0472	validation_1-rmse:11.0258
[33]	validation_0-rmse:10.0292	validation_1-rmse:11.0148
[34]	validation_0-rmse:10.015	validation_1-rmse:11.0045
[35]	validation_0-rmse:9.99029	validation_1-rmse:10.9972
[36]	validation_0-rmse:9.97531	validation_1-rmse:10.9881
[37]	validation_0-rmse:9.96311	validation_1-rmse:10.9793
[38]	validation_0-rmse:9.95032	validation_1-rmse:10.9736
[39]	validation_0-rmse:9.94061	validation_1-rmse:10.9693
[40]	validation_0-rmse:9.91308	validation_1-rmse:10.9703
[41]	validation_0-rmse:9.90264	validation_1-rmse:10.9662
[42]	validation_0-rmse:9.88789	validation_1-rmse:10.9628
[43]	validation_0-rmse:9.87412	validation_1-rmse:10.9585
[44]	validation_0-rmse:9.86663	validation_1-rmse:10.9579
[45]	validation_0-rmse:9.85868	validation_1-rmse:10.9577
[46]	validation_0-rmse:9.84059	validation_1-rmse:10.9535
[47]	validation_0-rmse:9.83144	validation_1-rmse:10.9487
[48]	validation_0-rmse:9.81289	validation_1-rmse:10.9484
[49]	validation_0-rmse:9.80588	validation_1-rmse:10.9464
[50]	validation_0-rmse:9.79913	validation_1-rmse:10.951
[51]	validation_0-rmse:9.77903	validation_1-rmse:10.9547
[52]	validation_0-rmse:9.77032	validation_1-rmse:10.9517
[53]	validation_0-rmse:9.76585	validation_1-rmse:10.9496
[54]	validation_0-rmse:9.75359	validation_1-rmse:10.9566
[55]	validation_0-rmse:9.73735	validation_1-rmse:10.9701
[56]	validation_0-rmse:9.72141	validation_1-rmse:10.9691
[57]	validation_0-rmse:9.70599	validation_1-rmse:10.9641
[58]	validation_0-rmse:9.7013	validation_1-rmse:10.9624
[59]	validation_0-rmse:9.69543	validation_1-rmse:10.9585
[60]	validation_0-rmse:9.68408	validation_1-rmse:10.9551
[61]	validation_0-rmse:9.6776	validation_1-rmse:10.9556
[62]	validation_0-rmse:9.67111	validation_1-rmse:10.9539
[63]	validation_0-rmse:9.66824	validation_1-rmse:10.9546
[64]	validation_0-rmse:9.66684	validation_1-rmse:10.9541
[65]	validation_0-rmse:9.66167	validation_1-rmse:10.9532
[66]	validation_0-rmse:9.65588	validation_1-rmse:10.9504
[67]	validation_0-rmse:9.64408	validation_1-rmse:10.9459
[68]	validation_0-rmse:9.62961	validation_1-rmse:10.951
[69]	validation_0-rmse:9.61842	validation_1-rmse:10.9463
[70]	validation_0-rmse:9.60465	validation_1-rmse:10.9479
[71]	validation_0-rmse:9.59596	validation_1-rmse:10.9449
[72]	validation_0-rmse:9.58831	validation_1-rmse:10.948
[73]	validation_0-rmse:9.58089	validation_1-rmse:10.9494
[74]	validation_0-rmse:9.57104	validation_1-rmse:10.9504
[75]	validation_0-rmse:9.56611	validation_1-rmse:10.9528
[76]	validation_0-rmse:9.5576	validation_1-rmse:10.9513
[77]	validation_0-rmse:9.55195	validation_1-rmse:10.9519
[78]	validation_0-rmse:9.54705	validation_1-rmse:10.9506
[79]	validation_0-rmse:9.54101	validation_1-rmse:10.9493
[80]	validation_0-rmse:9.53768	validation_1-rmse:10.9479
[81]	validation_0-rmse:9.52568	validation_1-rmse:10.9484
[82]	validation_0-rmse:9.52209	validation_1-rmse:10.9486
[83]	validation_0-rmse:9.51266	validation_1-rmse:10.9485
[84]	validation_0-rmse:9.5054	validation_1-rmse:10.947
[85]	validation_0-rmse:9.5015	validation_1-rmse:10.9489
[86]	validation_0-rmse:9.49348	validation_1-rmse:10.9445
[87]	validation_0-rmse:9.49084	validation_1-rmse:10.9442
[88]	validation_0-rmse:9.487	validation_1-rmse:10.9436
[89]	validation_0-rmse:9.48428	validation_1-rmse:10.9431
[90]	validation_0-rmse:9.47784	validation_1-rmse:10.9406
[91]	validation_0-rmse:9.46667	validation_1-rmse:10.9451
[92]	validation_0-rmse:9.45733	validation_1-rmse:10.9403
[93]	validation_0-rmse:9.45069	validation_1-rmse:10.9381
[94]	validation_0-rmse:9.44471	validation_1-rmse:10.9396
[95]	validation_0-rmse:9.44252	validation_1-rmse:10.9392
[96]	validation_0-rmse:9.44125	validation_1-rmse:10.9374
[97]	validation_0-rmse:9.43992	validation_1-rmse:10.9371
[98]	validation_0-rmse:9.42947	validation_1-rmse:10.9295
[99]	validation_0-rmse:9.42229	validation_1-rmse:10.9278
[100]	validation_0-rmse:9.41556	validation_1-rmse:10.9253
[101]	validation_0-rmse:9.40333	validation_1-rmse:10.9324
[102]	validation_0-rmse:9.39818	validation_1-rmse:10.9321
[103]	validation_0-rmse:9.39093	validation_1-rmse:10.9298
[104]	validation_0-rmse:9.38631	validation_1-rmse:10.9302
[105]	validation_0-rmse:9.3831	validation_1-rmse:10.9299
[106]	validation_0-rmse:9.38077	validation_1-rmse:10.9333
[107]	validation_0-rmse:9.37759	validation_1-rmse:10.9344
[108]	validation_0-rmse:9.37308	validation_1-rmse:10.9335
[109]	validation_0-rmse:9.36958	validation_1-rmse:10.9333
[110]	validation_0-rmse:9.36486	validation_1-rmse:10.9307
[111]	validation_0-rmse:9.35983	validation_1-rmse:10.9316
[112]	validation_0-rmse:9.35196	validation_1-rmse:10.9307
[113]	validation_0-rmse:9.34739	validation_1-rmse:10.9293
[114]	validation_0-rmse:9.34425	validation_1-rmse:10.9283
[115]	validation_0-rmse:9.34253	validation_1-rmse:10.9283
[116]	validation_0-rmse:9.332	validation_1-rmse:10.9258
[117]	validation_0-rmse:9.3252	validation_1-rmse:10.9252
[118]	validation_0-rmse:9.32092	validation_1-rmse:10.9229
[119]	validation_0-rmse:9.31966	validation_1-rmse:10.9225
[120]	validation_0-rmse:9.31627	validation_1-rmse:10.9198
[121]	validation_0-rmse:9.31394	validation_1-rmse:10.9199
[122]	validation_0-rmse:9.30973	validation_1-rmse:10.9212
[123]	validation_0-rmse:9.30291	validation_1-rmse:10.9222
[124]	validation_0-rmse:9.29176	validation_1-rmse:10.9189
[125]	validation_0-rmse:9.28774	validation_1-rmse:10.9166
[126]	validation_0-rmse:9.28659	validation_1-rmse:10.9169
[127]	validation_0-rmse:9.28083	validation_1-rmse:10.9159
[128]	validation_0-rmse:9.27546	validation_1-rmse:10.9126
[129]	validation_0-rmse:9.27281	validation_1-rmse:10.9128
[130]	validation_0-rmse:9.27143	validation_1-rmse:10.914
[131]	validation_0-rmse:9.26778	validation_1-rmse:10.9122
[132]	validation_0-rmse:9.26681	validation_1-rmse:10.9127
[133]	validation_0-rmse:9.26177	validation_1-rmse:10.9118
[134]	validation_0-rmse:9.25581	validation_1-rmse:10.9099
[135]	validation_0-rmse:9.24827	validation_1-rmse:10.9076
[136]	validation_0-rmse:9.2459	validation_1-rmse:10.907
[137]	validation_0-rmse:9.24408	validation_1-rmse:10.9073
[138]	validation_0-rmse:9.2402	validation_1-rmse:10.9068
[139]	validation_0-rmse:9.2381	validation_1-rmse:10.9048
[140]	validation_0-rmse:9.23374	validation_1-rmse:10.9107
[141]	validation_0-rmse:9.22918	validation_1-rmse:10.9116
[142]	validation_0-rmse:9.22118	validation_1-rmse:10.9068
[143]	validation_0-rmse:9.21622	validation_1-rmse:10.9074
[144]	validation_0-rmse:9.21048	validation_1-rmse:10.9072
[145]	validation_0-rmse:9.20567	validation_1-rmse:10.9164
[146]	validation_0-rmse:9.2027	validation_1-rmse:10.9171
[147]	validation_0-rmse:9.19919	validation_1-rmse:10.9168
[148]	validation_0-rmse:9.19179	validation_1-rmse:10.9124
[149]	validation_0-rmse:9.18661	validation_1-rmse:10.9115
[150]	validation_0-rmse:9.1842	validation_1-rmse:10.9125
[151]	validation_0-rmse:9.17725	validation_1-rmse:10.9128
[152]	validation_0-rmse:9.16982	validation_1-rmse:10.9092
[153]	validation_0-rmse:9.16417	validation_1-rmse:10.9072
[154]	validation_0-rmse:9.15879	validation_1-rmse:10.9072
[155]	validation_0-rmse:9.1565	validation_1-rmse:10.907
[156]	validation_0-rmse:9.15422	validation_1-rmse:10.9062
[157]	validation_0-rmse:9.14877	validation_1-rmse:10.9062
[158]	validation_0-rmse:9.14082	validation_1-rmse:10.9036
[159]	validation_0-rmse:9.13759	validation_1-rmse:10.906
[160]	validation_0-rmse:9.13315	validation_1-rmse:10.9132
[161]	validation_0-rmse:9.12658	validation_1-rmse:10.9131
[162]	validation_0-rmse:9.12619	validation_1-rmse:10.9132
[163]	validation_0-rmse:9.12357	validation_1-rmse:10.9125
[164]	validation_0-rmse:9.11836	validation_1-rmse:10.9126
[165]	validation_0-rmse:9.1144	validation_1-rmse:10.9109
[166]	validation_0-rmse:9.11063	validation_1-rmse:10.9081
[167]	validation_0-rmse:9.1065	validation_1-rmse:10.9102
[168]	validation_0-rmse:9.10257	validation_1-rmse:10.9101
[169]	validation_0-rmse:9.10046	validation_1-rmse:10.9103
[170]	validation_0-rmse:9.0987	validation_1-rmse:10.9109
[171]	validation_0-rmse:9.09039	validation_1-rmse:10.9067
[172]	validation_0-rmse:9.08953	validation_1-rmse:10.9072
[173]	validation_0-rmse:9.08526	validation_1-rmse:10.908
[174]	validation_0-rmse:9.08001	validation_1-rmse:10.9062
[175]	validation_0-rmse:9.0755	validation_1-rmse:10.9057
[176]	validation_0-rmse:9.07073	validation_1-rmse:10.9089
[177]	validation_0-rmse:9.06732	validation_1-rmse:10.9105
[178]	validation_0-rmse:9.06279	validation_1-rmse:10.9104
[179]	validation_0-rmse:9.05982	validation_1-rmse:10.9113
[180]	validation_0-rmse:9.05665	validation_1-rmse:10.9107
[181]	validation_0-rmse:9.05449	validation_1-rmse:10.9106
[182]	validation_0-rmse:9.05095	validation_1-rmse:10.913
[183]	validation_0-rmse:9.04997	validation_1-rmse:10.9127
[184]	validation_0-rmse:9.04881	validation_1-rmse:10.9124
[185]	validation_0-rmse:9.04305	validation_1-rmse:10.9106
[186]	validation_0-rmse:9.03763	validation_1-rmse:10.9108
[187]	validation_0-rmse:9.03181	validation_1-rmse:10.9072
[188]	validation_0-rmse:9.03032	validation_1-rmse:10.9139
[189]	validation_0-rmse:9.02846	validation_1-rmse:10.9135
[190]	validation_0-rmse:9.02839	validation_1-rmse:10.9135
[191]	validation_0-rmse:9.02415	validation_1-rmse:10.9133
[192]	validation_0-rmse:9.02409	validation_1-rmse:10.9133
[193]	validation_0-rmse:9.023	validation_1-rmse:10.9145
[194]	validation_0-rmse:9.01824	validation_1-rmse:10.9153
[195]	validation_0-rmse:9.01444	validation_1-rmse:10.9153
[196]	validation_0-rmse:9.00916	validation_1-rmse:10.918
[197]	validation_0-rmse:9.00524	validation_1-rmse:10.9172
[198]	validation_0-rmse:9.00003	validation_1-rmse:10.9167
[199]	validation_0-rmse:8.99855	validation_1-rmse:10.9154
[200]	validation_0-rmse:8.99699	validation_1-rmse:10.9155
[201]	validation_0-rmse:8.99312	validation_1-rmse:10.9164
[202]	validation_0-rmse:8.99307	validation_1-rmse:10.9164
[203]	validation_0-rmse:8.98909	validation_1-rmse:10.9166
[204]	validation_0-rmse:8.98611	validation_1-rmse:10.9174
[205]	validation_0-rmse:8.97956	validation_1-rmse:10.9165
[206]	validation_0-rmse:8.97928	validation_1-rmse:10.9166
[207]	validation_0-rmse:8.97672	validation_1-rmse:10.9163
[208]	validation_0-rmse:8.97362	validation_1-rmse:10.9154
Stopping. Best iteration:
[158]	validation_0-rmse:9.14082	validation_1-rmse:10.9036

{'colsample_bytree': 0.8, 'gamma': 1.8, 'learning_rate': 0, 'max_depth': 3, 'min_child_weight': 2.0, 'n_estimators': 0, 'nthread': 0, 'objective': 0, 'reg_alpha': 1.4000000000000001, 'reg_lambda': 2.4000000000000004, 'subsample': 0.9}
dict_keys(['colsample_bytree', 'gamma', 'learning_rate', 'max_depth', 'min_child_weight', 'n_estimators', 'nthread', 'objective', 'reg_alpha', 'reg_lambda', 'subsample'])



train error: 87.885
test error: 117.319
Hyperopt error:

In [113]:

In [113]: loss
Out[113]: 10.83521143739511

In [114]: cat nohup_final.out
train error: 117.181
test error: 126.047
Logistic Regression score: 548.124152155805
RandomForest feature importance: [  7.65536963e-03   2.22279256e-03   2.73438291e-02   7.16702316e-03
   6.56777671e-03   1.23560046e-02   5.70947675e-01   1.83405441e-02
   2.33281489e-02   1.26914006e-02   5.24451112e-03   7.93742996e-03
   4.93288094e-03   5.66244357e-03   4.88024057e-03   2.15871403e-03
   6.55386051e-03   1.58389863e-03   1.01665043e-03   7.47596381e-03
   0.00000000e+00   1.56660653e-03   5.93773465e-04   9.50713081e-03
   0.00000000e+00   1.34453710e-03   0.00000000e+00   2.83547819e-04
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   5.36322562e-04   0.00000000e+00   3.71375364e-03   2.38890738e-02
   4.61060880e-03   4.60436432e-03   2.95133471e-03   3.26616970e-03
   3.04084581e-03   3.02097624e-03   7.26143702e-04   7.79918228e-04
   5.87109862e-04   1.65359528e-01   5.99595951e-03   4.88632225e-03
   4.85975217e-03   1.78090631e-02]
Tested parameters: [{'max_depth': 8, 'n_estimators': 30}, {'max_depth': 8, 'n_estimators': 40}, {'max_depth': 8, 'n_estimators': 50}, {'max_depth': 8, 'n_estimators': 60}, {'max_depth': 8, 'n_estimators': 70}, {'max_depth': 8, 'n_estimators': 80}, {'max_depth': 8, 'n_estimators': 90}, {'max_depth': 8, 'n_estimators': 100}, {'max_depth': 9, 'n_estimators': 30}, {'max_depth': 9, 'n_estimators': 40}, {'max_depth': 9, 'n_estimators': 50}, {'max_depth': 9, 'n_estimators': 60}, {'max_depth': 9, 'n_estimators': 70}, {'max_depth': 9, 'n_estimators': 80}, {'max_depth': 9, 'n_estimators': 90}, {'max_depth': 9, 'n_estimators': 100}, {'max_depth': 10, 'n_estimators': 30}, {'max_depth': 10, 'n_estimators': 40}, {'max_depth': 10, 'n_estimators': 50}, {'max_depth': 10, 'n_estimators': 60}, {'max_depth': 10, 'n_estimators': 70}, {'max_depth': 10, 'n_estimators': 80}, {'max_depth': 10, 'n_estimators': 90}, {'max_depth': 10, 'n_estimators': 100}, {'max_depth': 11, 'n_estimators': 30}, {'max_depth': 11, 'n_estimators': 40}, {'max_depth': 11, 'n_estimators': 50}, {'max_depth': 11, 'n_estimators': 60}, {'max_depth': 11, 'n_estimators': 70}, {'max_depth': 11, 'n_estimators': 80}, {'max_depth': 11, 'n_estimators': 90}, {'max_depth': 11, 'n_estimators': 100}]
Best Estimator: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=11,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
Best Parameters: {'max_depth': 11, 'n_estimators': 100}
train error: 94.33
test error with Best Parameters: 121.52
test error without Best Parameters: 131.327
MSE: 122.6024
XGB MSE score: 122.21274670518012
XGBoost without Grid Search train error: 111.321
XGBoost without Grid Search test error: 122.213
Best Estimator: XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=8, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
Best Parameters: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 100}
Best Score: 0.59894293112934
['mean_fit_time', 'mean_score_time', 'mean_test_score', 'mean_train_score', 'param_learning_rate', 'param_max_depth', 'param_n_estimators', 'params', 'rank_test_score', 'split0_test_score', 'split0_train_score', 'split1_test_score', 'split1_train_score', 'split2_test_score', 'split2_train_score', 'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score']
XGBoost with Grid Search train error: 4.486
XGBoost with Grid Search test error: 4.629
Performance between XGBoost with and without Grid Search: +11758.4%
Traceback (most recent call last):
  File "exploration20180803.py", line 362, in <module>
    y_test=y_test
TypeError: regression_params_opt() got an unexpected keyword argument 'x_train'

In [115]: list_result_hyperopt
Out[115]:
[(10.974313284886879,
  {'best_iteration': 231,
   'colsample_bytree': 0.35000000000000003,
   'gamma': 3.2,
   'learning_rate': 0.1,
   'max_depth': 6,
   'min_child_weight': 1.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 1.5,
   'reg_lambda': 0.4,
   'subsample': 0.75}),
 (10.865129657610609,
  {'best_iteration': 222,
   'colsample_bytree': 0.35000000000000003,
   'gamma': 1.2000000000000002,
   'learning_rate': 0.1,
   'max_depth': 8,
   'min_child_weight': 5.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 2.6,
   'reg_lambda': 1.3,
   'subsample': 0.65}),
 (11.091083955897776,
  {'best_iteration': 60,
   'colsample_bytree': 0.9,
   'gamma': 0.6000000000000001,
   'learning_rate': 0.1,
   'max_depth': 6,
   'min_child_weight': 1.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 4.0,
   'reg_lambda': 2.6,
   'subsample': 0.65}),
 (11.07627453060895,
  {'best_iteration': 130,
   'colsample_bytree': 0.8500000000000001,
   'gamma': 2.6,
   'learning_rate': 0.1,
   'max_depth': 5,
   'min_child_weight': 2.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 0.7000000000000001,
   'reg_lambda': 2.0,
   'subsample': 0.65}),
 (10.83521143739511,
  {'colsample_bytree': 0.8,
   'gamma': 1.8,
   'learning_rate': 0.02,
   'max_depth': 8,
   'min_child_weight': 2.0,
   'n_estimators': 605,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 2.4000000000000004,
   'reg_lambda': 1.4000000000000001,
   'subsample': 0.9}),
 (11.00998626747346,
  {'best_iteration': 49,
   'colsample_bytree': 0.9500000000000001,
   'gamma': 3.8000000000000003,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 4.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 1.2000000000000002,
   'reg_lambda': 0.4,
   'subsample': 0.7000000000000001}),
 (10.840754497358439,
  {'best_iteration': 129,
   'colsample_bytree': 0.55,
   'gamma': 2.8000000000000003,
   'learning_rate': 0.1,
   'max_depth': 9,
   'min_child_weight': 4.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 1.2000000000000002,
   'reg_lambda': 2.6,
   'subsample': 0.7000000000000001}),
 (10.977118473913857,
  {'best_iteration': 68,
   'colsample_bytree': 0.9500000000000001,
   'gamma': 3.4000000000000004,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 3.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 0.0,
   'reg_lambda': 2.9000000000000004,
   'subsample': 0.8}),
 (10.973392475384443,
  {'best_iteration': 68,
   'colsample_bytree': 0.9500000000000001,
   'gamma': 1.0,
   'learning_rate': 0.1,
   'max_depth': 7,
   'min_child_weight': 4.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 2.3000000000000003,
   'reg_lambda': 3.0,
   'subsample': 0.8}),
 (10.91541734259072,
  {'best_iteration': 158,
   'colsample_bytree': 0.7000000000000001,
   'gamma': 2.4000000000000004,
   'learning_rate': 0.1,
   'max_depth': 8,
   'min_child_weight': 1.0,
   'n_estimators': 1500,
   'nthread': -1,
   'objective': 'reg:linear',
   'reg_alpha': 2.0,
   'reg_lambda': 1.2000000000000002,
   'subsample': 0.65})]

In [116]: loss
Out[116]: 10.83521143739511

In [117]: y
Out[117]:
0         25
1         32
2         29
3         22
4          6
5          9
6         31
7         21
8         18
9         26
10        21
11        11
12        24
13        21
14        26
15         6
16        18
17        12
18        45
19        15
20        19
21        15
22        32
23         3
24        26
25         8
26        14
27        15
28        17
29        22
          ..
252078     4
252079     7
252080     2
252081     8
252082     8
252083     7
252084     2
252085     4
252086     7
252087    11
252088     7
252089     8
252090     6
252091     4
252092     2
252093     2
252094     8
252095     8
252096     8
252097     4
252098     2
252099     8
252100     6
252101     2
252102     7
252103     6
252104     6
252105     7
252106     8
252107     5
Name: visitors, Length: 252108, dtype: int64

In [118]: np.mean(y)
Out[118]: 20.973761245180636

In [119]: np.mean((y-np.mean(y))**2)
Out[119]: 280.7961803300048

In [120]: np.sqrt(np.mean((y-np.mean(y))**2))
Out[120]: 16.756974080364415

In [121]:

In [121]:

In [121]: y_train
Out[121]:
107651    24
163691    44
92590     11
226442     8
28716     14
139342    17
24184     21
78564     19
34156     26
186975     5
228618    21
51795      8
200350    21
55685      9
81228     41
112227    18
228229    21
2010      29
32564      6
165237     7
124473    17
186348     2
111985     7
47559     20
190380    17
198074    22
171841    27
180422    49
61667     20
103188     9
          ..
184779    15
214176     3
235796     6
103355    92
5311      25
199041    45
64925     13
194027     5
59735     14
769       47
64820     25
67221      4
41090     26
16023     12
191335    21
175203    15
126324    13
112727    12
87498     15
168266     2
213458    25
137337    10
54886      1
207892     3
110268    50
119879    12
103694    17
131932     8
146867    27
121958     7
Name: visitors, Length: 201686, dtype: int64

In [122]: y_pred
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-122-5a040a10fabc> in <module>()
----> 1 y_pred

NameError: name 'y_pred' is not defined

In [123]: y_pred = xgb_hyperopt1.predict(X_train)



In [124]:

In [124]: y_pred
Out[124]:
array([ 23.0494194 ,  46.57117462,   9.23552799, ...,  12.14904976,
        28.92042732,   7.10402203], dtype=float32)

In [125]: mean_squared_error(y_train.values, y_pred)
Out[125]: 87.88543066300771

In [126]: mean_squared_error(y_train.values(), y_pred)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-126-59f34fdebee7> in <module>()
----> 1 mean_squared_error(y_train.values(), y_pred)

TypeError: 'numpy.ndarray' object is not callable

In [127]: mean_squared_error(y_train.values, y_pred)
Out[127]: 87.88543066300771

In [128]: y_pred
Out[128]:
array([ 23.0494194 ,  46.57117462,   9.23552799, ...,  12.14904976,
        28.92042732,   7.10402203], dtype=float32)

In [129]: y_train.values
Out[129]: array([24, 44, 11, ...,  8, 27,  7])

In [130]: np.mean((y_pred-y.train.values)**2)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-130-92b24f92176c> in <module>()
----> 1 np.mean((y_pred-y.train.values)**2)

~/python3/lib/python3.6/site-packages/pandas/core/generic.py in __getattr__(self, name)
   3079             if name in self._info_axis:
   3080                 return self[name]
-> 3081             return object.__getattribute__(self, name)
   3082
   3083     def __setattr__(self, name, value):

AttributeError: 'Series' object has no attribute 'train'

In [131]: np.mean((y_pred-y_train.values)**2)
Out[131]: 87.88543066300771

In [132]: np.sqrt(mean_squared_error(y_train.values, y_pred))
Out[132]: 9.374722964600485

In [133]: plt
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-133-775d6b1824e5> in <module>()
----> 1 plt

NameError: name 'plt' is not defined

In [134]: import matplotlib.pyplot as plt
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-134-eff513f636fd> in <module>()
----> 1 import matplotlib.pyplot as plt

~/python3/lib/python3.6/site-packages/matplotlib/pyplot.py in <module>()
    113
    114 from matplotlib.backends import pylab_setup
--> 115 _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
    116
    117 _IP_REGISTERED = None

~/python3/lib/python3.6/site-packages/matplotlib/backends/__init__.py in pylab_setup()
     30     # imports. 0 means only perform absolute imports.
     31     backend_mod = __import__(backend_name,
---> 32                              globals(),locals(),[backend_name],0)
     33
     34     # Things we pull in from all backends

~/python3/lib/python3.6/site-packages/matplotlib/backends/backend_macosx.py in <module>()
     17
     18 import matplotlib
---> 19 from matplotlib.backends import _macosx
     20
     21 from .backend_agg import RendererAgg, FigureCanvasAgg

RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.

In [135]: import matplotlib.pyplot as plt
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-135-eff513f636fd> in <module>()
----> 1 import matplotlib.pyplot as plt

~/python3/lib/python3.6/site-packages/matplotlib/pyplot.py in <module>()
    113
    114 from matplotlib.backends import pylab_setup
--> 115 _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
    116
    117 _IP_REGISTERED = None

~/python3/lib/python3.6/site-packages/matplotlib/backends/__init__.py in pylab_setup()
     30     # imports. 0 means only perform absolute imports.
     31     backend_mod = __import__(backend_name,
---> 32                              globals(),locals(),[backend_name],0)
     33
     34     # Things we pull in from all backends

~/python3/lib/python3.6/site-packages/matplotlib/backends/backend_macosx.py in <module>()
     17
     18 import matplotlib
---> 19 from matplotlib.backends import _macosx
     20
     21 from .backend_agg import RendererAgg, FigureCanvasAgg

RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.

In [136]: %matplotlib inline
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-136-e27d371d6baa> in <module>()
----> 1 get_ipython().magic('matplotlib inline')

~/python3/lib/python3.6/site-packages/IPython/core/interactiveshell.py in magic(self, arg_s)
   2144         magic_name, _, magic_arg_s = arg_s.partition(' ')
   2145         magic_name = magic_name.lstrip(prefilter.ESC_MAGIC)
-> 2146         return self.run_line_magic(magic_name, magic_arg_s)
   2147
   2148     #-------------------------------------------------------------------------

~/python3/lib/python3.6/site-packages/IPython/core/interactiveshell.py in run_line_magic(self, magic_name, line)
   2065                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
   2066             with self.builtin_trap:
-> 2067                 result = fn(*args,**kwargs)
   2068             return result
   2069

<decorator-gen-107> in matplotlib(self, line)

~/python3/lib/python3.6/site-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
    185     # but it's overkill for just that one bit of state.
    186     def magic_deco(arg):
--> 187         call = lambda f, *a, **k: f(*a, **k)
    188
    189         if callable(arg):

~/python3/lib/python3.6/site-packages/IPython/core/magics/pylab.py in matplotlib(self, line)
     97             print("Available matplotlib backends: %s" % backends_list)
     98         else:
---> 99             gui, backend = self.shell.enable_matplotlib(args.gui)
    100             self._show_matplotlib_backend(args.gui, backend)
    101

~/python3/lib/python3.6/site-packages/IPython/core/interactiveshell.py in enable_matplotlib(self, gui)
   2928                 gui, backend = pt.find_gui_and_backend(self.pylab_gui_select)
   2929
-> 2930         pt.activate_matplotlib(backend)
   2931         pt.configure_inline_support(self, backend)
   2932

~/python3/lib/python3.6/site-packages/IPython/core/pylabtools.py in activate_matplotlib(backend)
    304     matplotlib.rcParams['backend'] = backend
    305
--> 306     import matplotlib.pyplot
    307     matplotlib.pyplot.switch_backend(backend)
    308

~/python3/lib/python3.6/site-packages/matplotlib/pyplot.py in <module>()
    113
    114 from matplotlib.backends import pylab_setup
--> 115 _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
    116
    117 _IP_REGISTERED = None

~/python3/lib/python3.6/site-packages/matplotlib/backends/__init__.py in pylab_setup()
     30     # imports. 0 means only perform absolute imports.
     31     backend_mod = __import__(backend_name,
---> 32                              globals(),locals(),[backend_name],0)
     33
     34     # Things we pull in from all backends

~/python3/lib/python3.6/site-packages/matplotlib/backends/backend_macosx.py in <module>()
     17
     18 import matplotlib
---> 19 from matplotlib.backends import _macosx
     20
     21 from .backend_agg import RendererAgg, FigureCanvasAgg

RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.

In [137]:

In [137]:

In [137]: np.histogram
Out[137]: <function numpy.lib.function_base.histogram>

In [138]: np.histogram(y_pred,y_train)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-138-6aa2d7a88ef9> in <module>()
----> 1 np.histogram(y_pred,y_train)

~/python3/lib/python3.6/site-packages/numpy/lib/function_base.py in histogram(a, bins, range, normed, weights, density)
    786         if (np.diff(bins) < 0).any():
    787             raise ValueError(
--> 788                 'bins must increase monotonically.')
    789
    790         # Initialize empty histogram

ValueError: bins must increase monotonically.

In [139]: np.histogram(y_pred_y_train)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-139-edd87f485b22> in <module>()
----> 1 np.histogram(y_pred_y_train)

NameError: name 'y_pred_y_train' is not defined

In [140]: np.histogram(y_pred_y-train)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-140-f8b0cded6235> in <module>()
----> 1 np.histogram(y_pred_y-train)

NameError: name 'y_pred_y' is not defined

In [141]: np.histogram(y_pred-y_train)
Out[141]:
(array([     1,      3,      3,     11,     19,    621, 197566,   3458,
             3,      1]),
 array([-395.11645508, -336.08457947, -277.05270386, -218.02082825,
        -158.98895264,  -99.95707703,  -40.92520142,   18.10667419,
          77.1385498 ,  136.17042542,  195.20230103]))

In [142]: np.histogram(np.abs(y_pred-y_train),range(20))
Out[142]:
(array([25394, 24428, 22349, 20050, 17426, 14753, 12617, 10499,  8612,
         7192,  6105,  4962,  4198,  3589,  2970,  2565,  2075,  1698,  1466]),
 array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19]))

In [143]: np.histogram(np.abs(y_pred-y_train),range(100))
Out[143]:
(array([25394, 24428, 22349, 20050, 17426, 14753, 12617, 10499,  8612,
         7192,  6105,  4962,  4198,  3589,  2970,  2565,  2075,  1698,
         1466,  1187,  1053,   821,   709,   613,   506,   468,   399,
          369,   311,   228,   201,   192,   162,   166,   118,   120,
           96,    86,    83,    75,    58,    61,    54,    33,    44,
           46,    25,    30,    31,    19,    26,    16,    25,    21,
           17,    13,    18,     8,    18,    14,    11,     8,    11,
            9,     1,     8,     4,     5,    15,     7,     2,     6,
            5,     6,     4,     3,     3,     2,     3,     4,     5,
            6,     3,     4,     2,     3,     0,     3,     1,     3,
            2,     2,     0,     0,     1,     0,     2,     3,     1]),
 array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]))

In [144]: np.histogram(np.abs(y_pred-y_test),range(100))
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-144-1677c9069bfa> in <module>()
----> 1 np.histogram(np.abs(y_pred-y_test),range(100))

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in wrapper(left, right, name, na_op)
    719                 lvalues = lvalues.values
    720
--> 721         result = wrap_results(safe_na_op(lvalues, rvalues))
    722         return construct_result(
    723             left,

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in safe_na_op(lvalues, rvalues)
    680         try:
    681             with np.errstate(all='ignore'):
--> 682                 return na_op(lvalues, rvalues)
    683         except Exception:
    684             if isinstance(rvalues, ABCSeries):

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in na_op(x, y)
    656         try:
    657             result = expressions.evaluate(op, str_rep, x, y,
--> 658                                           raise_on_error=True, **eval_kwargs)
    659         except TypeError:
    660             if isinstance(y, (np.ndarray, ABCSeries, pd.Index)):

~/python3/lib/python3.6/site-packages/pandas/core/computation/expressions.py in evaluate(op, op_str, a, b, raise_on_error, use_numexpr, **eval_kwargs)
    209     if use_numexpr:
    210         return _evaluate(op, op_str, a, b, raise_on_error=raise_on_error,
--> 211                          **eval_kwargs)
    212     return _evaluate_standard(op, op_str, a, b, raise_on_error=raise_on_error)
    213

~/python3/lib/python3.6/site-packages/pandas/core/computation/expressions.py in _evaluate_standard(op, op_str, a, b, raise_on_error, **eval_kwargs)
     62         _store_test_result(False)
     63     with np.errstate(all='ignore'):
---> 64         return op(a, b)
     65
     66

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in <lambda>(x, y)
     93         rmul=arith_method(operator.mul, names('rmul'), op('*'),
     94                           default_axis=default_axis, reversed=True),
---> 95         rsub=arith_method(lambda x, y: y - x, names('rsub'), op('-'),
     96                           default_axis=default_axis, reversed=True),
     97         rtruediv=arith_method(lambda x, y: operator.truediv(y, x),

ValueError: operands could not be broadcast together with shapes (201686,) (50422,)

In [145]: history
%paste
pwd
cd Desktop/
ls
cd DataScience/
ls
cd Jordan/
ls
cd code/
ls
cd Datalab/
ls
cd Recruit\ Restaurant\ Visitor\ Forecasting
ls
%paste
%paste
%paste
%paste
%paste
print(rf.predict(y_test))
%paste
%paste
train_error_rf
test_error_rf
%paste
%paste
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, min_samples_leaf=10)
rf.fit(X_train, y_train)
%paste
%paste
%paste
%paste
%paste
%paste
train_error_rf
test_error_rf
print(rf.feature_importances_)
print(rf.predict(X_test))
%paste
%paste
%paste
%paste
%paste
%paste
%paste
test_error_xgb
%paste
perf = round((test_error_xgb/test_error_xgb_grid)*100, 2)
perf
perf = round((test_error_xgb/test_error_xgb_grid), 2)
perf
%paste
%paste
%paste
rf2 = RandomForestRegressor(params=mygcv.best_params_)
rf2 = RandomForestRegressor(params_clf=mygcv.best_params_)
mygcv.best_params_
%paste
X_train
%paste
%paste
y_test.shape
X_test.shape
X_train.shape
y_train.shape
y_test
y_test.columns
X_test.columns
X_test.columns == X_train.colums
X_test.columns == X_train.columns
df_result_hyperopt
%paste
%paste
%paste
aaa.keys()
%paste
cat data/hyperopt_output.csv
aaa
print(format('aaa'))
print(format(aaa))
print(format(aaa))
aaa
hyper_parametres
params
%paste
%paste
%paste
%paste
X_test.values
%paste
%paste
%paste
clear
aaa
list_result_hyperopt
list_result_hyperopt[0]
s = sorted(list_result_hyperopt)
s
zip(s)
list(zip(s))[0]
list(zip(*s))[0]
s[0]
clear
%paste
ps
ps
aaa
train[col]
y.shape
y.shape
%paste
loss
cat nohup_final.out
list_result_hyperopt
loss
y
np.mean(y)
np.mean((y-np.mean(y))**2)
np.sqrt(np.mean((y-np.mean(y))**2))
y_train
y_pred
y_pred = xgb_hyperopt1.predict(X_train)
y_pred
mean_squared_error(y_train.values, y_pred)
mean_squared_error(y_train.values(), y_pred)
mean_squared_error(y_train.values, y_pred)
y_pred
y_train.values
np.mean((y_pred-y.train.values)**2)
np.mean((y_pred-y_train.values)**2)
np.sqrt(mean_squared_error(y_train.values, y_pred))
plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
%matplotlib inline
np.histogram
np.histogram(y_pred,y_train)
np.histogram(y_pred_y_train)
np.histogram(y_pred_y-train)
np.histogram(y_pred-y_train)
np.histogram(np.abs(y_pred-y_train),range(20))
np.histogram(np.abs(y_pred-y_train),range(100))
np.histogram(np.abs(y_pred-y_test),range(100))
history

In [146]: np.histogram(np.abs(y_pred-y),range(100))
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-146-52affa01c7e5> in <module>()
----> 1 np.histogram(np.abs(y_pred-y),range(100))

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in wrapper(left, right, name, na_op)
    719                 lvalues = lvalues.values
    720
--> 721         result = wrap_results(safe_na_op(lvalues, rvalues))
    722         return construct_result(
    723             left,

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in safe_na_op(lvalues, rvalues)
    680         try:
    681             with np.errstate(all='ignore'):
--> 682                 return na_op(lvalues, rvalues)
    683         except Exception:
    684             if isinstance(rvalues, ABCSeries):

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in na_op(x, y)
    656         try:
    657             result = expressions.evaluate(op, str_rep, x, y,
--> 658                                           raise_on_error=True, **eval_kwargs)
    659         except TypeError:
    660             if isinstance(y, (np.ndarray, ABCSeries, pd.Index)):

~/python3/lib/python3.6/site-packages/pandas/core/computation/expressions.py in evaluate(op, op_str, a, b, raise_on_error, use_numexpr, **eval_kwargs)
    209     if use_numexpr:
    210         return _evaluate(op, op_str, a, b, raise_on_error=raise_on_error,
--> 211                          **eval_kwargs)
    212     return _evaluate_standard(op, op_str, a, b, raise_on_error=raise_on_error)
    213

~/python3/lib/python3.6/site-packages/pandas/core/computation/expressions.py in _evaluate_standard(op, op_str, a, b, raise_on_error, **eval_kwargs)
     62         _store_test_result(False)
     63     with np.errstate(all='ignore'):
---> 64         return op(a, b)
     65
     66

~/python3/lib/python3.6/site-packages/pandas/core/ops.py in <lambda>(x, y)
     93         rmul=arith_method(operator.mul, names('rmul'), op('*'),
     94                           default_axis=default_axis, reversed=True),
---> 95         rsub=arith_method(lambda x, y: y - x, names('rsub'), op('-'),
     96                           default_axis=default_axis, reversed=True),
     97         rtruediv=arith_method(lambda x, y: operator.truediv(y, x),

ValueError: operands could not be broadcast together with shapes (201686,) (252108,)

In [147]: np.mean(y)
Out[147]: 20.973761245180636

In [148]: np.mean(y_train)
Out[148]: 20.949104052834603

In [149]: np.mean(y_test)
Out[149]: 21.072389036531671

In [150]: git status
  File "<ipython-input-150-31fea9c68626>", line 1
    git status
             ^
SyntaxError: invalid syntax


In [151]: