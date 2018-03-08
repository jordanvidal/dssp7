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
    max_eval = 100
    trials = Trials()

    def rmse_score(params):
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
              'max_depth': hp.choice('max_depth', list(range(3, 12))),
              'min_child_weight': hp.quniform('min_child_weight', 1., 5., 1),
              'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
              'nthread': hp.choice('nthread', [8,]),
              'n_jobs': hp.choice('n_jobs', [8,]),
              'n_estimators': hp.choice('n_estimators', [10000]),
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
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

df_result_hyperopt_nicolas = pd.DataFrame(columns=[np.append('score', list(space4rf.keys()))])
df_result_hyperopt_nicolas.to_csv("data/output_nicolas.csv", sep=";")