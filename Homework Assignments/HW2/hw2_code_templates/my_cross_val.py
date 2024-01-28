import numpy as np

import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score

def my_cross_val(model, loss_func, X, y, k=10):

    kf = KFold(n_splits=k, random_state=None, shuffle=False)

    results = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        if loss_func == 'mse':
            test_err = mean_squared_error(y_pred, y_test)
        elif loss_func == 'err_rate':
            test_err = 1-accuracy_score(y_test, y_pred)
        else:
            raise ValueError('Unknown loss function:', loss_func)

        results.append(test_err)

    results = np.asarray(results)

    return results

