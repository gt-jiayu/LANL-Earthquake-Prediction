# Using lightgbm, xgboost, catboost and Neural Network to predict test set separately
# Stacking is used to combine these algorithm later,
# the first layer of stacking include lightgbm, xgboost, catboost and Neural Network
# the second layer of stacking is NuSVR
# Using the result of stacking as the final prediction

import os
import gc
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import NuSVR
from tensorflow import keras
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin

OPT_PARAMS = False  # weather to optimize parameters
# hyperopt parameters
NUM_EVALS = 1000  # hyperopt num evals
N_FOLDS = 8  # 8 fold cross validation
# lightGBM parameters
LGBM_MAX_LEAVES = 2 ** 10  # max leaves of lightgbm tree
LGBM_MAX_DEPTH = 25  # the max depth of lightgbm tree
EVAL_METRIC_LGBM_REG = 'mae'  # using mean absolute error as metric
# XGBoost parameters
XGB_MAX_LEAVES = 2 ** 10  # max leaves of xgboost tree
XGB_MAX_DEPTH = 25  # the max depth of xgboost tree
EVAL_METRIC_XGB_REG = 'mae'  # using mean absolute error as metric
# catboost parameters
CB_MAX_DEPTH = 8  # the max depth of catboost tree
OBJECTIVE_CB_REG = 'MAE'  # using mean absolute error as metric
OUTPUT_DIR = 'output'  # save files output by this code


def read_data():
    """
    read data from disk
    :return: train set features, train set labels, test set features
    """
    print("read_data")
    scaled_train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_train_X.csv'))
    train_y = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'))
    scaled_test_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'scaled_test_X.csv'))
    return scaled_train_X, train_y, scaled_test_X


def features_select(scaled_train_X, train_y, scaled_test_X):
    """
    eliminate features that Pearson coefficient above 0.05 to labels
    :param scaled_train_X: train set features
    :param train_y: train set labels
    :param scaled_test_X: test set features
    :return: train set and test set features after selection
    """
    print("features select")
    pcol = []
    pcor = []
    pval = []
    y = train_y['time_to_failure'].values

    # use pearson's to eliminate features with suspect correlation - helped kaggle scores
    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True)
    df.dropna(inplace=True)
    df = df.loc[df['pval'] <= 0.05]

    drop_cols = []

    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    return scaled_train_X, scaled_test_X


def cross_valid():
    """
    define k fold cross validation
    :return: K-Folds cross-validator
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    return kf


def quick_hyperopt(data, labels, kf, package='lgb', num_evals=NUM_EVALS, diagnostic=False):
    """
    using bayes optimise to tune parameters
    :param data: train set features
    :param labels: train set labels
    :param kf: K-Folds cross-validator
    :param package: algorithm
    :param num_evals: the number of fmin optimize
    :param diagnostic: diabnostic optimize progress or not
    :return: best parameters
    """
    # ==========================================
    # ================LightGBM==================
    # ==========================================
    if package == 'lgb':
        print('Running {} rounds of LightGBM parameter opt imisation:'.format(num_evals))
        gc.collect()  # release space

        integer_params = ['num_leaves', 'max_bin', 'min_data_in_leaf', 'bagging_freq']

        # the objective function for fmin
        def objective(space_params):
            # cast float parameters to int
            for integer_param in integer_params:
                space_params[integer_param] = int(space_params[integer_param])

            cv_results = lgb.cv(space_params, train, num_boost_round=60000, nfold=N_FOLDS,
                                stratified=False, early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG)
            best_loss = cv_results['l1-mean'][-1]

            return {'loss': best_loss, 'status': STATUS_OK}

        train = lgb.Dataset(data, labels)

        metric_list = ['mae', 'rmse']
        objective_list = ['mae', 'huber', 'gamma', 'fair', 'tweedie', 'regression']
        space = {
            'boosting': 'gbdt',
            'max_depth': -1,
            'num_leaves': hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
            'max_bin': hp.quniform('max_bin', 32, 255, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
            'min_sum_hessian_in_leaf': hp.quniform('min_sum_hessian_in_leaf', 0, 0.1, 0.001),
            'lambda_l1': hp.uniform('lambda_l1', 0, 5),
            'lambda_l2': hp.uniform('lambda_l2', 0, 5),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
            'metrci': hp.choice('metric', metric_list),
            'objective': hp.choice('objective', objective_list),
            'feature_fraction': hp.quniform('feature_fraction', 0.5, 1, 0.02),
            'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1, 0.02),
            'bagging_freq': hp.quniform('bagging_freq', 1, 20, 1),
            'random_state': 42
        }
        trials = Trials()
        # If the parameter form is list/array, fmin returns the index of the best value
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=num_evals, trials=trials)

        # Using index to get best value
        best['metric'] = metric_list[best['metric']]
        best['objective'] = objective_list[best['objective']]

        for integer_param in integer_params:
            best[integer_param] = int(best[integer_param])

        if diagnostic:
            return best, trials
        else:
            return best

    # ==========================================
    # ================XGBoost==================
    # ==========================================
    if package == 'xgb':
        print("Running {} rounds of XGBoost parameter optimisation.".format(num_evals))
        gc.collect()  # release space

        integer_params = ['max_depth']

        def objective(space_params):
            for integer_param in integer_params:
                space_params[integer_param] = int(space_params[integer_param])

            if space_params['tree_method']['tree_method'] == 'hist':
                space_params['max_bin'] = int(space_params['tree_method'].get('max_bin'))
                if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'deepwise':
                    space_params['grow_policy'] = space_params['tree_method'].get('grow_policy').get('grow_policy').get(
                        'grow_policy')
                    space_params['tree_method'] = 'hist'
                else:
                    space_params['grow_policy'] = 'lossguide'
                    space_params['max_leaves'] = int(
                        space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves'))
                    space_params['tree_method'] = 'hist'
            else:
                space_params['tree_method'] = space_params['tree_method'].get('tree_method')

            cv_results = xgb.cv(space_params, train, nfold=6, num_boost_round=1000,
                                early_stopping_rounds=100, metrics=EVAL_METRIC_XGB_REG, shuffle=False)
            best_loss = cv_results['test-mae-mean'].iloc[-1]
            return {'loss': best_loss, 'status': STATUS_OK}

        train = xgb.DMatrix(data, labels)

        boosting_list = ['gbtree', 'gblinear']
        metric_list = ['mae', "rmse"]
        tree_method = [{'tree_method': 'exact'},
                       {'tree_method': 'approx'},
                       {'tree_method': 'hist',
                        'max_bin': hp.quniform('max_bin', 16, 256, 1),
                        'grow_policy': {'grow_policy': {'grow_policy': 'deepwise'},
                                        'grow_policy': {'grow_policy': 'lossguide',
                                                        'max_leaves': hp.quniform('max_leaves', 32, XGB_MAX_LEAVES,
                                                                                  1)}}}]
        objective_list = ['reg:linear', 'reg:gamma', 'reg:tweedie']
        space = {
            'boosting': hp.choice('boosting', boosting_list),
            'tree_method': hp.choice('tree_method', tree_method),
            'max_depth': hp.quniform('max_depth', 2, XGB_MAX_DEPTH, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 5),
            'reg_lambda': hp.uniform('reg_lambda', 0, 5),
            'min_child_weight': hp.uniform('min_child_weight', 0, 5),
            'min_split_loss': hp.uniform('min_split_loss', 0, 5),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.5)),
            'eval_metric': hp.choice('eval_metric', metric_list),
            'objective': hp.choice('objective', objective_list),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.01),
            'colsample_bynode': hp.quniform('colsample_bynode', 0.1, 1, 0.01),
            'colsample_bylevel': hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'n_jobs': 4
        }

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=num_evals, trials=trials)

        # If the parameter form is list/array, fmin returns the index of the best value
        best['boosting'] = boosting_list[best['boosting']]
        best['tree_method'] = tree_method[best['tree_method']]['tree_method']
        best['eval_metric'] = metric_list[best['eval_metric']]
        best['objective'] = objective_list[best['objective']]

        for integer_param in integer_params:
            best[integer_param] = int(best[integer_param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        if 'max_bin' in best:
            best['max_bin'] = int(best['max_bin'])

        if diagnostic:
            return best, trials
        else:
            return best

    # ==========================================
    # ================CatBoost==================
    # ==========================================
    if package == 'cb':
        print('Running {} rounds of CatBoost parameter optimisation:'.format(num_evals))
        gc.collect()

        integer_params = ['depth',
                          'min_data_in_leaf',
                          'max_bin']

        def objective(space_params):
            for param in integer_params:
                space_params[param] = int(space_params[param])

            if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
                space_params['bagging_temperature'] = bagging_temp

            if space_params['grow_policy']['grow_policy'] == 'LossGuide':
                max_leaves = space_params['grow_policy'].get('max_leaves')
                space_params['max_leaves'] = int(max_leaves)

            space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
            space_params['grow_policy'] = space_params['grow_policy']['grow_policy']

            # random_strength >= 0
            space_params['random_strength'] = max(space_params['random_strength'], 0)
            # fold_len_multiplier >= 1
            space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)

            cv_results = cb.cv(train, space_params, fold_count=N_FOLDS,
                               early_stopping_rounds=25, stratified=False, partition_random_seed=42)

            best_loss = cv_results['test-MAE-mean'].iloc[-1]

            return {'loss': best_loss, 'status': STATUS_OK}

        train = cb.Pool(data, labels.astype('float32'))

        bootstrap_type = [{'bootstrap_type': 'Poisson'},
                          {'bootstrap_type': 'Bayesian',
                           'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                          {'bootstrap_type': 'Bernoulli'}]
        LEB = ['No', 'AnyImprovement', 'Armijo']
        grow_policy = [{'grow_policy': 'SymmetricTree'},
                       {'grow_policy': 'Depthwise'},
                       {'grow_policy': 'Lossguide',
                        'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]
        eval_metric_list = ['MAE', 'RMSE', 'Poisson']

        space = {'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),
                 'max_bin': hp.quniform('max_bin', 1, 32, 1),
                 'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 5),
                 'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),
                 'random_strength': hp.loguniform('random_strength', np.log(0.005), np.log(5)),
                 'bootstrap_type': hp.choice('bootstrap_type', bootstrap_type),
                 'learning_rate': hp.uniform('learning_rate', 0.05, 0.25),
                 'eval_metric': hp.choice('eval_metric', eval_metric_list),
                 'objective': OBJECTIVE_CB_REG,
                 'leaf_estimation_backtracking': hp.choice('leaf_estimation_backtracking', LEB),
                 'grow_policy': hp.choice('grow_policy', grow_policy),
                 'fold_len_multiplier': hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
                 'od_type': 'Iter',
                 'od_wait': 25,
                 'task_type': 'GPU',
                 'verbose': 0
                 }

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)

        # If the parameter form is list/array, fmin returns the index of the best value
        best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']
        best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']
        best['eval_metric'] = eval_metric_list[best['eval_metric']]
        best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]

        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])

        if diagnostic:
            return best, trials
        else:
            return best

    # ==========================================
    # =================sklearn==================
    # ==========================================
    if package == 'sklearn':
        print('Runing {} rounds for sklearn parameters optimisation'.format(num_evals))
        gc.collect()  # release space

        def objective(space_params):
            scores = []

            for fold_, (trn_idx, val_idx) in enumerate(kf.split(data, labels)):
                X_tr, X_val = data.iloc[trn_idx], data.iloc[val_idx]
                y_tr, y_val = labels.iloc[trn_idx], labels.iloc[val_idx]

                model = NuSVR(**space_params)
                model.fit(X_tr, y_tr)
                y_pred_val = model.predict(X_val).reshape(-1)
                score = mean_absolute_error(y_val, y_pred_val)
                scores.append(score)

            best_loss = np.mean(scores)
            return {'loss': best_loss, 'status': STATUS_OK}

        tol_list = [1e-5, 1e-3, 1e-4, 1e-2, 1e-1]
        space = {
            'nu': hp.uniform('nu', 1e-9, 1),
            'C': hp.loguniform('C', np.log(1e-3), np.log(1e2)),
            'kernel': 'rbf',
            'gamma': hp.uniform('gamma', 0, 1),
            'tol': hp.choice('tol', tol_list)
        }

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=num_evals, trials=trials)

        # If the parameter form is list/array, fmin returns the index of the best value
        best['tol'] = tol_list[best['tol']]

        if diagnostic:
            return best, trials
        else:
            return best


def create_model(input_dim=10):
    """
    function to create neural network model
    :param input_dim: input sample dimension
    :return: neural network model
    """
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, activation="relu", input_dim=input_dim))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(96, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1, activation="linear"))

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mae')
    return model


def train_model(scaled_train_X, scaled_test_X, train_y, kf, params=None,
                model_type='lgb', model=None):
    """
    train model to fit test set
    :param scaled_train_X: train set features
    :param scaled_test_X: test set features
    :param train_y: train set labels
    :param kf: K-Folds cross validators
    :param params: model parameters
    :param model_type: model type
    :param model: only for sklearn
    :return: prediction for train set with cross validation, prediction for test set
    """
    oof = np.zeros(len(scaled_train_X))
    prediction = np.zeros(len(scaled_test_X))
    feature_importance_df = pd.DataFrame()
    scores = []

    # To xgb, using stratified sample, so that no data leak
    # kf of xgb is different from others, so that the model fusion is better
    if model_type == 'xgb':
        kf = KFold(n_splits=6, shuffle=False, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(kf.split(scaled_train_X, train_y.values)):
        print('Fold {}:'.format(fold_))

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae', verbose=1000,
                      early_stopping_rounds=500)
            y_pred_val = model.predict(X_val, num_iteration=model.best_iteration_)
            y_pred = model.predict(scaled_test_X, num_iteration=model.best_iteration_)

        elif model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=scaled_train_X.columns)
            valid_data = xgb.DMatrix(data=X_val, label=y_val, feature_names=scaled_train_X.columns)
            watch_list = [(train_data, 'train'), (valid_data, 'valid')]
            model = xgb.train(dtrain=train_data, num_boost_round=1000, evals=watch_list,
                              early_stopping_rounds=100, verbose_eval=500, params=params)
            y_pred_val = model.predict(xgb.DMatrix(X_val, feature_names=scaled_train_X.columns),
                                       ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(scaled_test_X, feature_names=scaled_train_X.columns),
                                   ntree_limit=model.best_ntree_limit)

        elif model_type == 'sklearn':
            model = model
            model.fit(X_tr, y_tr)
            y_pred_tr = model.predict(X_tr).reshape(-1)
            y_pred_val = model.predict(X_val).reshape(-1)
            y_pred = model.predict(scaled_test_X).reshape(-1)
            train_l1 = mean_absolute_error(y_tr, y_pred_tr)
            score = mean_absolute_error(y_val, y_pred_val)
            print("Fold{0} : train's l1: {1:0.5f}, valid's l1: {2:0.5f}".format(fold_, train_l1, score))

        elif model_type == 'cb':
            model = cb.CatBoostRegressor(iterations=25000, eval_metric='MAE', early_stopping_rounds=500,
                                         task_type="CPU", **params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=[], use_best_model=True, verbose=False)
            y_pred_val = model.predict(X_val)
            y_pred = model.predict(scaled_test_X)
            train_l1 = model.best_score_['learn']['MAE']
            score = mean_absolute_error(y_val, y_pred_val)
            print("Fold{0} : train's l1: {1:0.5f}, valid's l1: {2:0.5f}".format(fold_, train_l1, score))

        else:  # nn
            call_ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                                                    verbose=1, mode='auto', baseline=None, restore_best_weights=True)
            model = create_model(X_tr.shape[-1])
            model.fit(X_tr, y_tr, epochs=500, batch_size=32, verbose=0, callbacks=[call_ES, ],
                      validation_data=[X_val, y_val])

            y_pred_val = model.predict(X_val)[:, 0]
            y_pred = model.predict(scaled_test_X)[:, 0]
            history = model.history.history
            tr_loss = history["loss"]
            val_loss = history["val_loss"]
            print("Fold{0} : train's l1: {1:0.5f}, valid's l1: {2:0.5f}".format(fold_, tr_loss, val_loss))

        oof[val_idx] = y_pred_val
        prediction = prediction + y_pred / kf.n_splits
        scores.append(mean_absolute_error(y_val, y_pred_val))

        if model_type == 'lgb':
            # features importance maybe useful for features selection
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = scaled_train_X.columns
            fold_importance_df['importance'] = model.feature_importances_
            fold_importance_df['fold'] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print("CV mean score: {0:0.5f}, std score: {1:0.5f}".format(np.mean(scores), np.std(scores)))
    return oof, prediction


def stack_model(kf, scaled_train_X, scaled_test_X, train_y):
    """
    Fused models with stacking
    :param kf: K-Folds cross-validator
    :param scaled_train_X: train set features
    :param scaled_test_X: test set features
    :param train_y: train set labels
    :return: None, output final submission
    """
    if OPT_PARAMS:
        lgb_params = quick_hyperopt(scaled_train_X, train_y, kf, package='lgb', num_evals=200)
        xgb_params = quick_hyperopt(scaled_train_X, train_y, kf, package='xgb', num_evals=100)
        cb_params = quick_hyperopt(scaled_train_X, train_y, kf, package='cb', num_evals=100)
        svr_params = quick_hyperopt(scaled_train_X, train_y, kf, package='sklearn', num_evals=100)
    else:
        lgb_params = {
            'num_leaves': 21,  # reducing to this helped with over fit
            'min_data_in_leaf': 20,
            'objective': 'regression',
            'max_depth': 108,  # bogus, check lgb hard limit max depth, depth controlled by min_data_in_leaf
            'learning_rate': 0.001,
            "boosting": "gbdt",
            "feature_fraction": 0.91,
            "bagging_freq": 1,
            "bagging_fraction": 0.91,
            "bagging_seed": 42,
            "metric": 'mae',
            "lambda_l1": 0.1,
            "verbosity": -1,
            "random_state": 42
        }
        xgb_params = {
            'colsample_bytree': 0.67,
            'eval_metric': 'mae',
            'learning_rate': 0.1,
            'max_depth': 6,
            'reg_lambda': 1.0,
            'subsample': 0.9
        }
        cb_params = {
            'objective': "MAE",
            'loss_function': 'MAE',
            'boosting_type': "Ordered"
        }
        svr_params = {
            'C': 15.844733356886566,
            'gamma': 0.0011238961587171446,
            'nu': 0.7631978191886737,
            'tol': 0.1
        }
    oof_lgb, prediction_lgb = train_model(scaled_train_X, scaled_test_X, train_y,
                                          kf, params=lgb_params, model_type='lgb')
    oof_xgb, prediction_xgb = train_model(scaled_train_X, scaled_test_X, train_y,
                                          kf, params=xgb_params, model_type='xgb')
    oof_cb, prediction_cb = train_model(scaled_train_X, scaled_test_X, train_y,
                                        kf, params=cb_params, model_type='cb')
    oof_nn, prediction_nn = train_model(scaled_train_X, scaled_test_X, train_y,
                                        kf, model_type='nn')

    # the first layer
    train_X = pd.DataFrame(index=range(oof_lgb.shape[0]))
    train_X['oof_lgb'] = oof_lgb['time_to_failure']
    train_X['oof_xgb'] = oof_xgb['time_to_failure']
    train_X['oof_cb'] = oof_cb['time_to_failure']
    train_X['oof_nn'] = oof_nn['time_to_failure']
    test_X = pd.DataFrame(index=range(prediction_lgb.shape[0]))
    test_X['prediction_lgb'] = prediction_lgb['time_to_failure']
    test_X['prediction_xgb'] = prediction_xgb['time_to_failure']
    test_X['prediction_cb'] = prediction_cb['time_to_failure']
    test_X['prediction_nn'] = prediction_nn['time_to_failure']

    # the second layer
    model = NuSVR(**svr_params)
    oof_svr, prediction_svr = train_model(train_X, test_X, train_y, kf, model_type='sklearn', model=model)
    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
    submission['time_to_failure'] = prediction_svr
    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_stacking.csv'), index=True)


if __name__ == "__main__":
    scaled_train_X, train_y, scaled_test_X = read_data()
    scaled_train_X, scaled_test_X = features_select(scaled_train_X, train_y, scaled_test_X)
    kf = cross_valid()
    stack_model(kf, scaled_train_X, scaled_test_X, train_y)
