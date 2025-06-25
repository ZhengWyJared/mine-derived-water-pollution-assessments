import pandas as pd
import numpy as np
import torch
import cvxpy as cp
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
import time
import pickle

def precompute_piecewise_vectors(
    features1, num_intervals_list, max_diff_list, device, 
    features2=None
):
    if features2 is None:
        num_samples = features1.shape[0]
        pair_indices = list(combinations(range(num_samples), 2))
    else:
        pair_indices = [(k, l) for k in range(features1.shape[0]) for l in range(features2.shape[0])]

    num_features = features1.shape[1]
    all_piecewise_vectors = []
    for i in range(num_features):
        max_diff = max_diff_list[i]
        intervals = np.linspace(0, max_diff, num_intervals_list[i])
        vecs = []
        for k, l in pair_indices:
            diff = (features1[l, i] - features1[k, i]) if features2 is None else (features2[l, i] - features1[k, i])
            abs_diff = abs(diff)
            v = torch.zeros(num_intervals_list[i] - 1, device=device)
            for seg in range(1, num_intervals_list[i]):
                if abs_diff < intervals[seg]:
                    v[seg - 1] = (abs_diff - intervals[seg - 1]) / (intervals[seg] - intervals[seg - 1])
                    break
                elif seg < num_intervals_list[i] - 1:
                    v[seg - 1] = 1
                else:
                    v[seg - 1] = 1
            vecs.append(v * torch.sign(diff))
        all_piecewise_vectors.append(torch.stack(vecs))
    return all_piecewise_vectors

def scoring_function(precomputed_vectors, num_intervals_list, delta_u, device):
    terms = []
    for i in range(len(num_intervals_list)):
        terms.append(torch.mv(precomputed_vectors[i], delta_u[i]))
    return sum(terms)

def build_label_for_pairs(labels_1d):
    y_list, pair_list = [], []
    n = len(labels_1d)
    for k in range(n):
        for l in range(k+1, n):
            v =  1. if labels_1d[l] > labels_1d[k] else -1. if labels_1d[l] < labels_1d[k] else 0.
            y_list.append(v)
            pair_list.append((k, l))
    return np.array(y_list, dtype=np.float32), pair_list


def solve_qp_cvxpy(precomputed_vectors, labels_for_pairs,
                   num_intervals_list, monotonic_info,
                   C1, J, device=torch.device('cpu')):
    num_features = len(num_intervals_list)
    t_p, t_I = [], []
    for v in labels_for_pairs:
        if v > 0:      t_p.append(J);  t_I.append(0.)
        elif v < 0:    t_p.append(-J); t_I.append(0.)
        else:          t_p.append(0.); t_I.append(J)
    t_p = np.array(t_p, dtype=np.float32)
    t_I = np.array(t_I, dtype=np.float32)

    pre_np = [pv.cpu().numpy() for pv in precomputed_vectors]
    lambda_z = C1 * num_features

    delta_vars, z_vars, exprs, cons = [], [], [], []
    for j in range(num_features):
        Mj = num_intervals_list[j] - 1
        dj = cp.Variable(Mj); delta_vars.append(dj)
        if monotonic_info[j] == 'inc':
            cons.append(dj >= 0); exprs.append(cp.sum(dj))
        else:
            cons.append(dj <= 0); exprs.append(-cp.sum(dj))

    p_terms = []
    for j in range(num_features):
        A = cp.Parameter(pre_np[j].shape, value=pre_np[j])
        pj = A @ delta_vars[j]; p_terms.append(pj)
        zj = cp.Variable(pj.shape, nonneg=True); z_vars.append(zj)
        cons += [zj >= pj, zj >= -pj]

    p_expr = sum(p_terms)
    abs_sum = sum(z_vars)
    I_expr = num_features - abs_sum
    for e in exprs:
        cons.append(e <= 1)

    loss = (cp.sum_squares(p_expr - t_p)
            + cp.sum_squares(I_expr - t_I)
            + lambda_z * cp.sum(abs_sum))
    prob = cp.Problem(cp.Minimize(loss), cons)
    prob.solve(
        solver=cp.OSQP,
        verbose=False,
        ignore_dpp=True,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=10000
    )

    delta_opt, diffs = [], []
    for dj, zj, pj in zip(delta_vars, z_vars, p_terms):
        if dj.value is None:
            raise ValueError("No solution.")
        delta_opt.append(dj.value)
        diffs.append(np.max(np.abs(zj.value - np.abs(pj.value))))
    return [torch.tensor(d, dtype=torch.float32, device=device) for d in delta_opt], diffs

def predict_with_confidence(feats_tr, labels_tr, feats_te,
                            delta_u_opt, num_intervals_list,
                            max_diff_list, J, device=torch.device('cpu')):
    ftr = torch.tensor(feats_tr, device=device)
    fte = torch.tensor(feats_te, device=device)
    pre_te = precompute_piecewise_vectors(ftr, num_intervals_list, max_diff_list, device, features2=fte)
    scores = scoring_function(pre_te, num_intervals_list, delta_u_opt, device).cpu().numpy()
    n_tr, n_te = feats_tr.shape[0], feats_te.shape[0]
    mat = scores.reshape(n_tr, n_te)
    uniq = np.unique(labels_tr)
    preds=[]
    for x in range(n_te):
        confs=[]
        for h in uniq:
            num=den=0.
            for k, lb in enumerate(labels_tr):
                if lb==h: continue
                den+=1
                pkx=mat[k,x]
                if lb>h and pkx>0:    num+= pkx/J
                elif lb<h and pkx<0:  num+= -pkx/J
            confs.append(num/den if den>0 else 0.)
        preds.append(uniq[np.argmax(confs)])
    return np.array(preds, dtype=np.float32)

def train_val_fold_once(feats_tr, labels_tr, feats_va, labels_va,
                        num_intervals, C1, monotonic_info,
                        max_diff_list, device):
    nf = feats_tr.shape[1]
    ni_list = [2 if len(np.unique(feats_tr[:,j]))==2 else num_intervals for j in range(nf)]
    pre_tr = precompute_piecewise_vectors(torch.tensor(feats_tr, device=device),
                                         ni_list, max_diff_list, device)
    y_pairs,_ = build_label_for_pairs(labels_tr)
    J = float(nf)
    delta_u_opt, diffs = solve_qp_cvxpy(pre_tr, y_pairs, ni_list, monotonic_info, C1, J, device)
    pred_va = predict_with_confidence(feats_tr, labels_tr, feats_va,
                                      delta_u_opt, ni_list, max_diff_list, J, device)
    return np.mean(pred_va==labels_va), delta_u_opt, ni_list, diffs

def run_cv_search_and_test(features, labels, max_diff_list,
                           interval_candidates, C1_candidates,
                           monotonic_info, n_splits, test_ratio,
                           device, random_state):
    #-- split
    tr_idx, te_idx = train_test_split(
        np.arange(len(labels)), test_size=test_ratio,
        stratify=labels, random_state=random_state)
    ftr, ltr = features[tr_idx], labels[tr_idx]
    fte, lte = features[te_idx], labels[te_idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = [(ni,c1) for ni in interval_candidates for c1 in C1_candidates]

    def eval_param(ni, c1):
        scores=[]
        for idx_tr, idx_va in skf.split(ftr, ltr):
            acc, *_ = train_val_fold_once(
                ftr[idx_tr], ltr[idx_tr],
                ftr[idx_va], ltr[idx_va],
                ni, c1, monotonic_info,
                max_diff_list, device)
            scores.append(acc)
        return np.mean(scores)

    cvs = Parallel(n_jobs=-1)(delayed(eval_param)(ni,c1) for ni,c1 in grid)
    best_ni, best_c1, best_cv = max(zip([g[0] for g in grid],
                                        [g[1] for g in grid], cvs),
                                   key=lambda x: x[2])

    # final train+test
    acc_va, delta_u_opt, ni_list, diffs = train_val_fold_once(
        ftr, ltr, fte, lte,
        best_ni, best_c1, monotonic_info,
        max_diff_list, device)

    pred_tr = predict_with_confidence(ftr, ltr, ftr, delta_u_opt, ni_list, max_diff_list, float(ftr.shape[1]), device)
    pred_te = predict_with_confidence(ftr, ltr, fte, delta_u_opt, ni_list, max_diff_list, float(ftr.shape[1]), device)

    return {
        'best_params':      (best_ni, best_c1, best_cv),
        'train_acc':        np.mean(pred_tr==ltr),
        'test_acc':         np.mean(pred_te==lte),
        'train_preds':      pred_tr,
        'test_preds':       pred_te,
        'train_labels':     ltr,
        'test_labels':      lte
    }

if __name__ == "__main__":
    # load data
    df = pd.read_csv(r"D:/JORS/LEV.csv")
    labels = df.iloc[:,0].to_numpy().astype(np.float32)
    features = df.iloc[:,1:].to_numpy().astype(np.float32)

    max_diff_list = [
        float(df.iloc[:,1+i].max() - df.iloc[:,1+i].min())
        for i in range(features.shape[1])
    ]
    monotonic_info = ['inc'] * features.shape[1]

    all_runs = []
    train_accs = []; test_accs = []
    train_f1s = [];  test_f1s = []
    best_params_list = []

    for i in range(5):
        res = run_cv_search_and_test(
            features, labels, max_diff_list,
            interval_candidates=[2,3,4,5],
            C1_candidates=[5, 6, 7,8,9, 10, 11],
            monotonic_info=monotonic_info,
            n_splits=5, test_ratio=0.2,
            device=torch.device('cpu'),
            random_state=i
        )
        best_params_list.append(res['best_params'])
        train_accs.append(res['train_acc'])
        test_accs.append(res['test_acc'])
        train_f1s.append(f1_score(res['train_labels'], res['train_preds'], average='macro'))
        test_f1s.append( f1_score(res['test_labels'],  res['test_preds'],  average='macro'))
        all_runs.append(res)

    summary = {
        'train_accs': train_accs,   'test_accs': test_accs,
        'train_f1s':  train_f1s,    'test_f1s':  test_f1s,
        'best_params': best_params_list,
        'mean_train_acc': np.mean(train_accs),
        'mean_test_acc':  np.mean(test_accs),
        'mean_train_f1':  np.mean(train_f1s),
        'mean_test_f1':   np.mean(test_f1s),
    }

    with open("final_results_5splits_LEV.pkl", "wb") as f:
        pickle.dump({'runs': all_runs, 'summary': summary}, f)

with open("final_results_5splits_ERA.pkl", "rb") as f:
    data = pickle.load(f)


summary = data['summary']
runs    = data['runs']

test_accs = summary['test_accs']
test_f1s  = summary['test_f1s']

mean_test_acc = np.mean(test_accs)
std_test_acc  = np.std(test_accs)

mean_test_f1  = np.mean(test_f1s)
std_test_f1   = np.std(test_f1s)

print("=== Summary ===")
print(f"Train ACCs       : {summary['train_accs']}")
print(f"Test  ACCs       : {test_accs}")
print(f"Train F1s        : {summary['train_f1s']}")
print(f"Test  F1s        : {test_f1s}")
print(f"Best params per run: {summary['best_params']}")
print(f"Mean train ACC   : {summary['mean_train_acc']:.4f}")
print(f"Mean test  ACC   : {mean_test_acc:.4f} ± {std_test_acc:.4f}")
print(f"Mean train F1    : {summary['mean_train_f1']:.4f}")
print(f"Mean test  F1    : {mean_test_f1:.4f} ± {std_test_f1:.4f}")
print()

# print results
for i, run in enumerate(runs, 1):
    bp = run['best_params']
    ta = run['train_acc']
    te = run['test_acc']
    print(f"--- Run {i} ---")
    print(f"Best params    : num_intervals={bp[0]}, C1={bp[1]}, CV_score={bp[2]:.4f}")
    print(f"Train   ACC    : {ta:.4f}")
    print(f"Test    ACC    : {te:.4f}")
    print()