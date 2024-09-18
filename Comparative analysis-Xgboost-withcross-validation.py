import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import time
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


data_path = r'DataAfterPreprocessing.xlsx'
data = pd.read_excel(data_path)


X = data.iloc[:, 0:3]
y = data.iloc[:, 4]
e = 0

performance_metrics = []
relationship_comparisons = []
start_time = time.time()  


param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


xgb_model = XGBRegressor(objective='reg:squarederror')


random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  
    cv=5,  
    scoring='neg_mean_squared_error',
    verbose=0
)


for i in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=16, test_size=5, random_state=i + 3)


    random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    

    best_model = random_search.best_estimator_
    print(f"{i+1} best parameter：", random_search.best_params_)

    y_pred = best_model.predict(X_test)
    
   
    d = 0
    n = len(y_test)
    total_pairs = n * (n - 1) / 2
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y_test.iloc[i] - y_test.iloc[j]) > e:
                if y_test.iloc[i] > y_test.iloc[j]:
                    real_relation = 'P'
                else:
                    real_relation = '-P'
            else:
                real_relation = 'I'

            if abs(y_pred[i] - y_pred[j]) > e:
                if y_pred[i] > y_pred[j]:
                    pred_relation = 'P'
                else:
                    pred_relation = '-P'
            else:
                pred_relation = 'I'

            if real_relation != pred_relation:
                d += 1

    performance_metric = 1 - d / total_pairs
    performance_metrics.append(performance_metric)


y_pred_full = best_model.predict(X)


d = 0
n = len(y)
total_pairs = n * (n - 1) / 2
relationship_comparisons_full = []
for i in range(n):
    for j in range(i + 1, n):
        if abs(y.iloc[i] - y.iloc[j]) > e:
            if y.iloc[i] > y.iloc[j]:
                real_relation = 'P'
            else:
                real_relation = '-P'
        else:
            real_relation = 'I'

        if abs(y_pred_full[i] - y_pred_full[j]) > e:
            if y_pred_full[i] > y_pred_full[j]:
                pred_relation = 'P'
            else:
                pred_relation = '-P'
        else:
            pred_relation = 'I'

        relationship_comparisons_full.append((real_relation, pred_relation))

        if real_relation != pred_relation:
            d += 1

performance_metric_full = 1 - d / total_pairs
end_time = time.time() 
elapsed_time = end_time - start_time


print("average performance：", np.mean(performance_metrics))

print("time：", elapsed_time)



