import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import mord
import pickle
from sklearn.svm import LinearSVC


df = pd.read_csv("D:/JORS/LEV.csv")
labels_raw = df.iloc[:, 0].astype(int)
features = df.iloc[:, 1:].astype(float)


le = LabelEncoder()
labels = le.fit_transform(labels_raw)


train_accs, test_accs = [], []
train_f1s,  test_f1s  = [], []
best_params_list, all_runs = [], []


param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0]  
}


for i in range(5):
    tr_idx, te_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2,
        stratify=labels, random_state=i
    )
    X_tr, y_tr = features.iloc[tr_idx], labels[tr_idx]
    X_te, y_te = features.iloc[te_idx], labels[te_idx]


    grid = GridSearchCV(
    estimator=LinearSVC(dual=False, max_iter=10000),
    param_grid={'C': [1.0, 0.1, 0.01]},
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
    grid.fit(X_tr, y_tr)

    best_model = grid.best_estimator_
    best_params = grid.best_params_


    pred_tr = best_model.predict(X_tr)
    pred_te = best_model.predict(X_te)
    pred_tr_raw = le.inverse_transform(pred_tr)
    pred_te_raw = le.inverse_transform(pred_te)
    y_tr_raw = le.inverse_transform(y_tr)
    y_te_raw = le.inverse_transform(y_te)

    train_acc = np.mean(pred_tr_raw == y_tr_raw)
    test_acc  = np.mean(pred_te_raw == y_te_raw)
    train_f1  = f1_score(y_tr_raw, pred_tr_raw, average='macro')
    test_f1   = f1_score(y_te_raw, pred_te_raw, average='macro')

    train_accs.append(train_acc);  test_accs.append(test_acc)
    train_f1s.append(train_f1);    test_f1s.append(test_f1)
    best_params_list.append(best_params)
    all_runs.append({
        'train_preds': pred_tr_raw,
        'test_preds':  pred_te_raw,
        'train_labels': y_tr_raw,
        'test_labels':  y_te_raw,
        'train_acc': train_acc,
        'test_acc':  test_acc,
        'train_f1':  train_f1,
        'test_f1':   test_f1,
        'best_params': best_params
    })


summary = {
    'train_accs': train_accs,
    'test_accs':  test_accs,
    'train_f1s':  train_f1s,
    'test_f1s':   test_f1s,
    'best_params': best_params_list,
    'mean_train_acc': np.mean(train_accs),
    'mean_test_acc':  np.mean(test_accs),
    'mean_train_f1':  np.mean(train_f1s),
    'mean_test_f1':   np.mean(test_f1s),
}
with open("final_results_5splits_LinearOrdinalSVM_LEV.pkl", "wb") as f:
    pickle.dump({'runs': all_runs, 'summary': summary}, f)


print(f"Train ACCs          : {summary['train_accs']}")
print(f"Test  ACCs          : {summary['test_accs']}")
print(f"Train F1s           : {summary['train_f1s']}")
print(f"Test  F1s           : {summary['test_f1s']}")
print(f"Best params per run : {summary['best_params']}")
print(f"Mean train ACC      : {summary['mean_train_acc']:.4f}")
print(f"Mean test  ACC      : {summary['mean_test_acc']:.4f} ± {np.std(test_accs):.4f}")
print(f"Mean train F1       : {summary['mean_train_f1']:.4f}")
print(f"Mean test  F1       : {summary['mean_test_f1']:.4f} ± {np.std(test_f1s):.4f}")
