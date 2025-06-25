import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC


file_path = 'D:/JORS/StructuralData.xlsx'
data = pd.read_excel(file_path)



X = data.iloc[:, [1, 2, 4]] 
y = data.iloc[:, 5]          
e = 0  


performance_metrics = []
relationship_comparisons = []
start_time = time.time()  


for iteration in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=16, test_size=5, random_state=iteration + 100)

    model = SVC(kernel='linear')

    X_train_pairs = []
    y_train_labels = []

    n_train = len(y_train)
    for i in range(n_train):
        for j in range(i + 1, n_train):
            X_train_pairs.append(X_train.iloc[i] - X_train.iloc[j])
   
            if abs(y_train.iloc[i] - y_train.iloc[j]) <= e:
                y_train_labels.append(2)
            elif y_train.iloc[i] > y_train.iloc[j]:
                y_train_labels.append(1)
            elif y_train.iloc[i] < y_train.iloc[j]:
                y_train_labels.append(0)

    X_train_pairs = np.array(X_train_pairs)
    y_train_labels = np.array(y_train_labels)


    model.fit(X_train_pairs, y_train_labels)


    X_test_pairs = []
    y_test_labels = []

    n_test = len(y_test)
    for i in range(n_test):
        for j in range(i + 1, n_test):
            X_test_pairs.append(X_test.iloc[i] - X_test.iloc[j])
            if abs(y_test.iloc[i] - y_test.iloc[j]) <= e:
                y_test_labels.append(2)
            elif y_test.iloc[i] > y_test.iloc[j]:
                y_test_labels.append(1)
            elif y_test.iloc[i] < y_test.iloc[j]:
                y_test_labels.append(0)

    X_test_pairs = np.array(X_test_pairs)
    y_test_labels = np.array(y_test_labels)

    y_pred = model.predict(X_test_pairs)

    d = sum(y_test_labels != y_pred)
    performance_metric = 1 - d / len(y_test_labels)
    performance_metrics.append(performance_metric)

print("mean performance：", np.mean(performance_metrics))

end_time = time.time()
elapsed_time = end_time - start_time
print("time：", elapsed_time)
pd.DataFrame(performance_metrics).to_csv(r'D:\JORS\performance_metrics_SVM.csv', index=False, header=False)

plt.figure(figsize=(10, 5))
plt.plot(range(len(performance_metrics)), performance_metrics, color='blue', label='Performance Metrics', linestyle='-', marker='o', alpha=0.5)


plt.xlabel('Iterations')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()
