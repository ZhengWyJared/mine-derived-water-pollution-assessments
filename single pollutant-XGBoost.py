import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import numpy as np
import time
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


file_path = 'D:/JORS/StructuralData.xlsx'
data = pd.read_excel(file_path)



X = data.iloc[:, [1, 2, 4]]  
y = data.iloc[:, 5]          
e = 0  


performance_metrics = []
start_time = time.time()


param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


xgb_model = XGBClassifier(objective='multi:softmax', num_class=3,
                          use_label_encoder=False, eval_metric='mlogloss')

random_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1  
)


for iteration in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=16, test_size=5, random_state=iteration + 100)

    X_train_pairs = []
    y_train_labels = []
    for i in range(len(y_train)):
        for j in range(i + 1, len(y_train)):
            diff = X_train.iloc[i] - X_train.iloc[j]
            X_train_pairs.append(diff)
            delta = y_train.iloc[i] - y_train.iloc[j]
            if abs(delta) <= e:
                y_train_labels.append(2)  
            elif delta > 0:
                y_train_labels.append(1)  
            else:
                y_train_labels.append(0)  

    X_train_pairs = np.array(X_train_pairs)
    y_train_labels = np.array(y_train_labels)


    random_search.fit(X_train_pairs, y_train_labels)
    best_model = random_search.best_estimator_
    print(f"best CV parameter for {iteration+1}：", random_search.best_params_)

    X_test_pairs = []
    y_test_labels = []
    for i in range(len(y_test)):
        for j in range(i + 1, len(y_test)):
            diff = X_test.iloc[i] - X_test.iloc[j]
            X_test_pairs.append(diff)
            delta = y_test.iloc[i] - y_test.iloc[j]
            if abs(delta) <= e:
                y_test_labels.append(2)
            elif delta > 0:
                y_test_labels.append(1)
            else:
                y_test_labels.append(0)

    X_test_pairs = np.array(X_test_pairs)
    y_test_labels = np.array(y_test_labels)

    y_pred = best_model.predict(X_test_pairs)
    d = np.sum(y_pred != y_test_labels)
    acc = 1 - d / len(y_test_labels)
    performance_metrics.append(acc)


print("mean performance：", np.mean(performance_metrics))
print("time：", time.time() - start_time)
pd.DataFrame(performance_metrics).to_csv(r'D:\JORS\performance_metrics.csv', index=False, header=False)