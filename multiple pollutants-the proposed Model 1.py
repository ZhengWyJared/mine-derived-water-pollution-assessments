import numpy as np
import pandas as pd
import torch
import cvxpy as cp
from itertools import combinations
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import matplotlib.pyplot as plt


def precompute_piecewise_vectors(features1, num_intervals_list, max_diff_list, device, features2=None):
    if features2 is None:
        pair_indices = list(combinations(range(features1.shape[0]), 2))
    else:
        pair_indices = [(k, l) for k in range(features1.shape[0]) for l in range(features2.shape[0])]
    F = features1.shape[1]
    all_vecs = []
    for j in range(F):
        intervals = np.linspace(0, max_diff_list[j], num_intervals_list[j])
        vecs = []
        for k, l in pair_indices:
            diff = (features1[l,j] - features1[k,j]) if features2 is None else (features2[l,j] - features1[k,j])
            abs_d = abs(diff).item()
            v = torch.zeros(num_intervals_list[j]-1, device=device)
            for seg in range(1, num_intervals_list[j]):
                if abs_d < intervals[seg]:
                    v[seg-1] = (abs_d - intervals[seg-1])/(intervals[seg] - intervals[seg-1])
                    break
                else:
                    v[seg-1] = 1.0
            v *= torch.sign(diff)
            vecs.append(v)
        all_vecs.append(torch.stack(vecs))
    return all_vecs

def scoring_function(precomputed_vectors, num_intervals_list, delta_u, device):
    return sum(torch.mv(precomputed_vectors[j], delta_u[j]) for j in range(len(num_intervals_list)))

def predict_with_confidence(feats_tr, labels_tr, feats_te,
                            delta_u, num_intervals_list, max_diff_list,
                            J, device):
    tr_t = torch.tensor(feats_tr, device=device)
    te_t = torch.tensor(feats_te, device=device)
    pre = precompute_piecewise_vectors(tr_t, num_intervals_list, max_diff_list, device, te_t)
    p_all = scoring_function(pre, num_intervals_list, delta_u, device).cpu().numpy()
    n_tr, n_te = feats_tr.shape[0], feats_te.shape[0]
    p_mat = p_all.reshape(n_tr, n_te)
    uniq = np.unique(labels_tr)
    preds = []
    for x in range(n_te):
        confs = []
        for h in uniq:
            num = denom = 0.0
            for k in range(n_tr):
                if labels_tr[k]==h: continue
                denom += 1
                pkx = p_mat[k,x]
                if labels_tr[k]>h and pkx>0:    num += pkx/J
                elif labels_tr[k]<h and pkx<0:  num += -pkx/J
            confs.append(num/denom if denom>0 else 0.0)
        preds.append(uniq[np.argmax(confs)])
    return np.array(preds, dtype=np.float32)

def generate_pairwise_targets(labels_mat, F):
    pair_list, P, I, R = [], [], [], []
    n = labels_mat.shape[0]
    for k in range(n):
        for l in range(k+1, n):
            a=b=c=0
            for col in range(labels_mat.shape[1]):
                ci, cj = labels_mat[k,col], labels_mat[l,col]
                d = abs(ci-cj)
                if d>0:
                    if ci>cj: a+=1
                    else:      b+=1
                else:
                    c+=1
            if a+c>0:
                if a>b:
                    Pij, Iij, Rij = (a-b)/(a+c), c/(a+c), b/(a+c)
                else:
                    denom = b+c if (b+c)>0 else 1
                    Pij, Iij, Rij = (a-b)/denom, c/denom, a/denom
            else:
                Pij=Iij=Rij=0.0
            pair_list.append((k,l))
            P.append(Pij*F)
            I.append(Iij*F)
            R.append(Rij*F)
    return pair_list, np.array(P, dtype=np.float32), np.array(I, dtype=np.float32), np.array(R, dtype=np.float32)

def build_label_for_pairs(labels_1d):
    y, pair_list = [], []
    n = len(labels_1d)
    for k in range(n):
        for l in range(k+1, n):
            if labels_1d[l]>labels_1d[k]:      y.append(1.0)
            elif labels_1d[l]<labels_1d[k]:    y.append(-1.0)
            else:                              y.append(0.0)
            pair_list.append((k,l))
    return np.array(y, dtype=np.float32), pair_list


def solve_qp_cvxpy(pre, t_p, t_I, num_intervals_list, mono, C1, C2, device):
    F = len(num_intervals_list)
    pre_np = [pv.cpu().numpy() for pv in pre]
    lambda_z = C1 * F

    delta_vars, z_vars, range_exprs = [], [], []
    constraints = []

    for j in range(F):
        Mj = num_intervals_list[j] - 1
        dj = cp.Variable(Mj)
        delta_vars.append(dj)

        if mono[j] == 'inc':
            constraints.append(dj >= 0)
            expr = cp.sum(dj)
        elif mono[j] == 'dec':
            constraints.append(dj <= 0)
            expr = -cp.sum(dj)
        else:
            expr = cp.norm1(dj)
        range_exprs.append(expr)

    p_terms = []
    for j in range(F):
        A_j = cp.Parameter(shape=pre_np[j].shape, value=pre_np[j])
        p_j = A_j @ delta_vars[j]
        p_terms.append(p_j)

        z_j = cp.Variable(p_j.shape, nonneg=True)
        z_vars.append(z_j)
        constraints += [z_j >= p_j, z_j >= -p_j]

    p_expr       = sum(p_terms)
    abs_sum_expr = sum(z_vars)
    I_expr       = F - abs_sum_expr

    for expr in range_exprs:
        constraints.append(expr <= 1)

    data_loss = (
        cp.sum_squares(p_expr - t_p)
      + cp.sum_squares(I_expr - t_I)
      + lambda_z * cp.sum(abs_sum_expr)
    )
    reg = C2 * sum(cp.sum_squares(dj) for dj in delta_vars)

    prob = cp.Problem(cp.Minimize(data_loss + reg), constraints)
    prob.solve(
        solver=cp.OSQP,
        verbose=False,
        ignore_dpp=True,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=10000
    )

    out = []
    for dj in delta_vars:
        if dj.value is None:
            raise ValueError("Infeasible!")
        out.append(torch.tensor(dj.value, dtype=torch.float32, device=device))
    return out



if __name__ == "__main__":
    df = pd.read_excel('D:/JORS/StructuralData.xlsx', engine='openpyxl')
    features   = df.iloc[:, [1,2,4]].to_numpy().astype(np.float32)
    labels_mat = df.iloc[:, 5:9].to_numpy().astype(np.float32)
    n, F = features.shape

    mean_val = labels_mat.mean(axis=1)
    q1,q2,q3 = np.percentile(mean_val, [25,50,75])
    labels_q = np.digitize(mean_val, bins=[q1,q2,q3]) + 1

    max_diff_list  = [(df.iloc[:,c].max()-df.iloc[:,c].min()) for c in [1,2,4]]
    monotonic_info = ['inc','dec','inc']
    J = float(F)

    interval_candidates = [2,3,4,5,6,7,8]
    C1_candidates = [1,5,10,15,20,22,25]
    C2_candidates = [0.01,0.1, 0.3,0.4,0.5, 1, 10, 20]

    loo = LeaveOneOut()
    acc1_tr, acc1_te = [], []
    acc2_tr, acc2_te = [], []
    param_history1, param_history2 = [], []

    for train_idx, test_idx in loo.split(features):
        Xtr, Xte         = features[train_idx], features[test_idx]
        Ltr_mat, Lte_mat = labels_mat[train_idx], labels_mat[test_idx]
        Ltr_q, Lte_q     = labels_q[train_idx], labels_q[test_idx]


        best1 = (-1, None)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for ni in interval_candidates:
            for C1 in C1_candidates:
                for C2 in C2_candidates:
                    cvs = []
                    for it_tr, it_va in inner.split(Xtr, Ltr_q):
                        X1, X2 = Xtr[it_tr], Xtr[it_va]
                        LM1    = Ltr_mat[it_tr]
                        LQ1, LQ2 = Ltr_q[it_tr], Ltr_q[it_va]

                        pre = precompute_piecewise_vectors(
                            torch.tensor(X1), [ni]*F, max_diff_list, torch.device('cpu')
                        )
                        _, P1, I1, _ = generate_pairwise_targets(LM1, F)
                        delta1 = solve_qp_cvxpy(
                            pre, P1, I1, [ni]*F,
                            monotonic_info, C1, C2, torch.device('cpu')
                        )
                        p_val = predict_with_confidence(
                            X1, LQ1, X2, delta1, [ni]*F, max_diff_list, J, torch.device('cpu')
                        )
                        cvs.append(np.mean(p_val==LQ2))
                    m = np.mean(cvs)
                    if m > best1[0]:
                        best1 = (m, (ni, C1, C2))

        ni1, C11, C21 = best1[1]
        param_history1.append((ni1, C11, C21))


        pre1 = precompute_piecewise_vectors(
            torch.tensor(Xtr), [ni1]*F, max_diff_list, torch.device('cpu')
        )
        _, P_tr1, I_tr1, _ = generate_pairwise_targets(Ltr_mat, F)
        delta1 = solve_qp_cvxpy(
            pre1, P_tr1, I_tr1, [ni1]*F,
            monotonic_info, C11, C21, torch.device('cpu')
        )
        p1_tr = predict_with_confidence(Xtr, Ltr_q, Xtr, delta1, [ni1]*F, max_diff_list, J, torch.device('cpu'))
        p1_te = predict_with_confidence(Xtr, Ltr_q, Xte, delta1, [ni1]*F, max_diff_list, J, torch.device('cpu'))
        acc1_tr.append(np.mean(p1_tr==Ltr_q))
        acc1_te.append(np.mean(p1_te==Lte_q))

        best2 = (-1, None)
        for ni in interval_candidates:
            for C1 in C1_candidates:
                for C2 in C2_candidates:
                    cvs = []
                    for it_tr, it_va in inner.split(Xtr, Ltr_q):
                        X1, X2 = Xtr[it_tr], Xtr[it_va]
                        LQ1, LQ2 = Ltr_q[it_tr], Ltr_q[it_va]
                        y1, _ = build_label_for_pairs(LQ1)
                        tp = np.where(y1>0, J, np.where(y1<0, -J, 0.0))
                        tI = np.where(y1==0, J, 0.0)

                        pre = precompute_piecewise_vectors(
                            torch.tensor(X1), [ni]*F, max_diff_list, torch.device('cpu')
                        )
                        delta2 = solve_qp_cvxpy(
                            pre, tp, tI, [ni]*F,
                            monotonic_info, C1, C2, torch.device('cpu')
                        )
                        p_val = predict_with_confidence(
                            X1, LQ1, X2, delta2, [ni]*F, max_diff_list, J, torch.device('cpu')
                        )
                        cvs.append(np.mean(p_val==LQ2))
                    m = np.mean(cvs)
                    if m > best2[0]:
                        best2 = (m, (ni, C1, C2))

        ni2, C12, C22 = best2[1]
        param_history2.append((ni2, C12, C22))

        y_full, _ = build_label_for_pairs(Ltr_q)
        tp2 = np.where(y_full>0, J, np.where(y_full<0, -J, 0.0))
        tI2 = np.where(y_full==0, J, 0.0)
        pre2 = precompute_piecewise_vectors(
            torch.tensor(Xtr), [ni2]*F, max_diff_list, torch.device('cpu')
        )
        delta2 = solve_qp_cvxpy(
            pre2, tp2, tI2, [ni2]*F,
            monotonic_info, C12, C22, torch.device('cpu')
        )
        p2_tr = predict_with_confidence(Xtr, Ltr_q, Xtr, delta2, [ni2]*F, max_diff_list, J, torch.device('cpu'))
        p2_te = predict_with_confidence(Xtr, Ltr_q, Xte, delta2, [ni2]*F, max_diff_list, J, torch.device('cpu'))
        acc2_tr.append(np.mean(p2_tr==Ltr_q))
        acc2_te.append(np.mean(p2_te==Lte_q))

    print(f"scheme 1 LOO average train_acc = {np.mean(acc1_tr):.4f}, test_acc = {np.mean(acc1_te):.4f}")
    print(f"scheme 2 LOO average train_acc = {np.mean(acc2_tr):.4f}, test_acc = {np.mean(acc2_te):.4f}\n")

    print("Best hyperparameter：")
    for i, ((ni1,C11,C21), (ni2,C12,C22)) in enumerate(zip(param_history1, param_history2), 1):
        print(f"Fold {i}:")
        print(f"  Scheme1 = (intervals={ni1}, C1={C11}, C2={C21})")
        print(f"  Scheme2 = (intervals={ni2}, C1={C12}, C2={C22})")

    plt.figure(figsize=(6,4))
    plt.plot(acc1_te, '-o', label='Scheme1 Test')
    plt.plot(acc2_te, '-s', label='Scheme2 Test')
    plt.xlabel('LOO Fold'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)
    plt.show()
