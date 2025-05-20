import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import matplotlib.pyplot as plt


file_path = r'DataAfterPreprocessing.xlsx'
data = pd.read_excel(file_path)


X = data.iloc[:, 0:3]
y = data.iloc[:, 4]
e=0

sum_weights = np.zeros(3)
sum_intercept = 0
performance_metrics = []
relationship_comparisons = []
start_time = time.time() 


for i in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=16, test_size=5, random_state=i+3) 


    model = LinearRegression()


    model.fit(X_train, y_train)

    sum_weights += model.coef_
    sum_intercept += model.intercept_


    y_pred = model.predict(X_test)

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


avg_weights = sum_weights / 100
avg_intercept = sum_intercept / 100


y_pred_full = np.dot(X, avg_weights) + avg_intercept

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



plt.figure(figsize=(10, 5))
plt.plot(range(len(y)), y, color='blue', label='Actual Values', linestyle='-', marker='o', alpha=0.5)
plt.plot(range(len(y_pred_full)), y_pred_full, color='red', label='Predicted Values', linestyle='--', marker='o', alpha=0.5)
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.xticks(ticks=range(len(y)))
plt.show()
