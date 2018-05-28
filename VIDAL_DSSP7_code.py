#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jordan VIDAL
DSSP7
2018-03-12

Recruit Restaurant Visitor Forecasting
Predict how many future visitors a restaurant will receive

"""

# Import from the standard library
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import preprocessing, ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename='report/output.log', level=logging.DEBUG,
                    format='%(asctime)s :: %(levelname)s :: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

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
for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])

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
# Label encoder pour les air_genre_name and air_area_name
lbl = preprocessing.LabelEncoder()
for i in range(10):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

# creation of train and test by merging train/test and stores dataset
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
#START MODELING

logging.info('END CLEANING&FE + START MODELING...')
y = train.pop('visitors')
X_train, X_test, y_train, y_test = train_test_split(train[col], y, test_size=0.2, random_state=42)


logging.info('1. Linear Regression')

linreg = LinearRegression().fit(X_train, y_train)
print("LinearRegression training set score: ", round(linreg.score(X_train, y_train), 4))
print("LinearRegression test score: ", round(linreg.score(X_test, y_test), 4))
linreg_train_rmse = round(np.sqrt(mean_squared_error(y_train, linreg.predict(X_train))), 4)
linreg_test_rmse = round(np.sqrt(mean_squared_error(y_test, linreg.predict(X_test))), 4)
# print("LinearRegression training RMSE: {}".format(train_error_lm))
print("LinearRegression RMSE: ", linreg_test_rmse)

scores = cross_val_score(linreg, X_train, y_train, cv=5)
print("Cross-validated scores:", scores)
y_predictions = linreg.predict(X_test)
print("First 10 Predictions", y_predictions[0:10])
print("First 10 Real Values", y_test[0:10])

logging.info('2. Ridge Regression')
ridge = Ridge().fit(X_train, y_train)
print("RidgeRegression training score: ", round(ridge.score(X_train, y_train), 4))
print("RidgeRegression test score: ", round(ridge.score(X_test, y_test), 4))
ridge_train_rmse = round(np.sqrt(mean_squared_error(y_train, ridge.predict(X_train))), 4)
ridge_test_rmse = round(np.sqrt(mean_squared_error(y_test, ridge.predict(X_test))), 4)
# print("Ridge training RMSE", ridge_train_rmse)
print("Ridge test RMSE", ridge_test_rmse)


logging.info('3. Lasso Regression')
lasso = Lasso().fit(X_train, y_train)
print("Lasso training score: ", round(lasso.score(X_train, y_train), 4))
print("Lasso test score: ", round(lasso.score(X_test, y_test), 4))
print("number of features used :", np.sum(lasso.coef_ != 0))
lasso_train_rmse = round(np.sqrt(mean_squared_error(y_train, lasso.predict(X_train))), 4)
lasso_test_rmse = round(np.sqrt(mean_squared_error(y_test, lasso.predict(X_test))), 4)
# print("Lasso training RMSE", lasso_train_rmse)
print("Lasso test RMSE", lasso_test_rmse)

logging.info('4. RandomForest without optimized parameters')

rf = RandomForestRegressor(n_jobs=-1)
rf.fit(X_train, y_train)
rf.predict(X_test)
print("RandomForest training score: ", round(rf.score(X_train, y_train), 4))
print("RandomForest test score: ", round(rf.score(X_test, y_test), 4))
rf_train_rmse = round(np.sqrt(mean_squared_error(y_train, rf.predict(X_train))), 4)
rf_test_rmse = round(np.sqrt(mean_squared_error(y_test, rf.predict(X_test))), 4)
importances = rf.feature_importances_
print("RandomForest feature importance", format(rf.feature_importances_))
print("RandomForest PREDICT : ", rf.predict(X_test))
print("RandomForest RMSE", rf_test_rmse)

std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
dtrain = xgboost.DMatrix(train[col], label=y)

logging.info('5. RandomForest - GridSearch')

params = dict(max_depth=list(range(8, 12)), n_estimators=list(range(30, 110, 10)))
mygcv = GridSearchCV(rf, param_grid=params).fit(X_train, y_train)
print("RF Tested parameters:", format(mygcv.cv_results_['params']))
print("RandomForest Best Estimator: ", format(mygcv.best_estimator_))
print("RandomForest Best Parameters: ", format(mygcv.best_params_))

logging.info('6. RandomForest with GridSearch parameters')

rf2 = RandomForestRegressor(max_depth=11, n_estimators=90, random_state=42)
rf2.fit(X_train, y_train)
rf2.predict(X_test)
print("RandomForest GridSearch training score: ", round(rf2.score(X_train, y_train)), 4)
print("RandomForest GridSearch test score: ", round(rf2.score(X_test, y_test)), 4)
rf2_train_rmse = round(np.sqrt(mean_squared_error(y_train, rf2.predict(X_train))), 4)
rf2_test_rmse = round(np.sqrt(mean_squared_error(y_test, rf2.predict(X_test))), 4)
print("RandomForest RMSE with GridSearch: {}".format(rf2_test_rmse))
print("RandomForest RMSE without GridSearch: {}".format(rf_test_rmse))

logging.info('7. XGBoost')

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb.predict(X_test)
print("XGB training score: ", round(xgb.score(X_train, y_train), 4))
print("XGB test score: ", round(xgb.score(X_test, y_test), 4))
xgb_train_rmse = round(np.sqrt(mean_squared_error(y_train, xgb.predict(X_train))), 4)
xgb_test_rmse = round(np.sqrt(mean_squared_error(y_test, xgb.predict(X_test))), 4)
# print("XGB test RMSE", xgb_train_rmse)
# print("XGB test RMSE score: {}".format(xgb_test_rmse))
# print("XGBoost RMSE without Grid Search train error: {}".format(xgb_train_rmse))
print("XGBoost test RMSE : ", xgb_test_rmse)

logging.info('7.1 XGBoost Grid Search')
params = dict(max_depth=list(range(5, 10)), n_estimators=[100], learning_rate=[0.1])
grid_search = GridSearchCV(xgb, param_grid=params, n_jobs=4).fit(X_train, y_train)
logging.info('7.2 End XGBoost Grid Search')
# summarize the results of the grid search
print('Best Estimator: {}'.format(grid_search.best_estimator_))
print('Best Parameters: {}'.format(grid_search.best_params_))
print('Best Score: {}'.format(grid_search.best_score_))
print(sorted(grid_search.cv_results_.keys()))

logging.info('7.3 XGBoost with Grid Search parameters')

params = grid_search.best_params_
xgb_grid = XGBRegressor(**params)
xgb_grid.fit(X_train, y_train)
print("XGB with GridSearch training score: ", round(xgb_grid.score(X_train, y_train), 4))
print("XGB with GridSearch test score: ", round(xgb_grid.score(X_test, y_test), 4))
xgb_grid_train_rmse = round(np.sqrt(mean_squared_error(y_train, xgb_grid.predict(X_train))), 4)
xgb_grid_test_rmse = round(np.sqrt(mean_squared_error(y_test, xgb_grid.predict(X_test))), 4)
# print("XGBoost with Grid Search train error: {}".format(train_error_xgb_grid))
print("XGBoost with Grid Search RMSE : {}".format(xgb_grid_test_rmse))
perf = round((xgb_grid.score(X_test, y_test) - xgb.score(X_test, y_test))*100, 2)
print("Performance between XGBoost with and without Grid Search: +{}%".format(perf))

logging.info('7.4 XGBoost Hyperopt')

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
    max_eval = 100
    trials = Trials()

    def rmse_score(params):
        model_fit = origin_model_d[current_model][0](
            **params).fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test),],early_stopping_rounds=50)
        y_pred_train = model_fit.predict(X_test)
        loss = np.sqrt(mean_squared_error(y_test, y_pred_train))**0.5
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

loss, params = sorted(list_result_hyperopt)[0]

logging.info('7.5. XGBoost with hyperopt parameters')
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

best_iteration = params['best_iteration']
del params['best_iteration']

params['n_estimators'] = best_iteration*5
params['learning_rate'] = 0.02

xgb_hyperopt1 = XGBRegressor(**params)
xgb_hyperopt1.fit(X_train, y_train)
xgbpred1 = xgb_hyperopt1.predict(X_test)
train_error_xgb_hyperopt1 = round(xgb_hyperopt1.score(X_train, y_train), 4)
test_error_xgb_hyperopt1 = round(xgb_hyperopt1.score(X_test, y_test), 4)
xgb_hyperopt1_train_rmse = round(np.sqrt(mean_squared_error(y_train, xgb_hyperopt1.predict(X_train))), 4)
xgb_hyperopt1_test_rmse = round(np.sqrt(mean_squared_error(y_test, xgb_hyperopt1.predict(X_test))), 4)

print("XGBoost with Hyperopt train error: {}".format(train_error_xgb_hyperopt1))
print("XGBoost with Hyperopt test error: {}".format(test_error_xgb_hyperopt1))
print("XGBoost with Hyperopt RMSE: {}".format(xgb_hyperopt1_test_rmse))
print("Hyperopt error: ", loss)