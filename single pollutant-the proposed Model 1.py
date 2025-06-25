import pandas as pd
import numpy as np
import torch
import cvxpy as cp
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed
import math
import time
import pickle
import os
from collections import Counter  
import matplotlib.pyplot as plt

def precompute_piecewise_vectors(
    features1, num_intervals_list, max_diff_list, device, 
    features2=None
):
    if features2 is None:
        pair_indices = list(combinations(range(features1.shape[0]), 2))
    else:
        pair_indices = [(k, l) for k in range(features1.shape[0]) for l in range(features2.shape[0])]

    num_features = features1.shape[1]
    all_vecs = []
    for i in range(num_features):
        max_diff = max_diff_list[i]
        intervals = np.linspace(0, max_diff, num_intervals_list[i])
        vecs = []
        for k, l in pair_indices:
            diff = (features1[l, i] - features1[k, i]
                    if features2 is None
                    else features2[l, i] - features1[k, i])
            abs_diff = abs(diff)
            v = torch.zeros(num_intervals_list[i] - 1, device=device)
            for seg in range(1, num_intervals_list[i]):
                if abs_diff < intervals[seg]:
                    v[seg - 1] = (abs_diff - intervals[seg - 1]) / (intervals[seg] - intervals[seg - 1])
                    break
                else:
                    v[seg - 1] = 1
            v = v * torch.sign(diff)
            vecs.append(v)
        all_vecs.append(torch.stack(vecs))
    return all_vecs

def scoring_function(precomputed_vectors, delta_u):
    marginals = []
    for j, pv in enumerate(precomputed_vectors):
        marginals.append(torch.mv(pv, delta_u[j]))
    p_vec = sum(marginals)
    return p_vec, marginals

errorlimit = 0
def build_label_for_pairs(labels_1d):
    n = len(labels_1d)
    y_list, pair_list = [], []
    for k in range(n):
        for l in range(k+1, n):
            diff = labels_1d[l] - labels_1d[k]
            if diff >  errorlimit:
                y_list.append(1.0)
            elif diff < -errorlimit:
                y_list.append(-1.0)
            else:
                y_list.append(0.0)
            pair_list.append((k, l))
    return np.array(y_list, dtype=np.float32), pair_list

def solve_qp_cvxpy(
    precomputed_vectors, labels_for_pairs,
    num_intervals_list, monotonic_info,
    C1, J,
    device=torch.device('cpu')
):
    num_features = len(num_intervals_list)


    t_p, t_I = [], []
    for val in labels_for_pairs:
        if   val > 0:
            t_p.append( J); t_I.append(0.)
        elif val < 0:
            t_p.append(-J); t_I.append(0.)
        else:
            t_p.append( 0.); t_I.append(J)
    t_p = np.array(t_p, dtype=np.float32)
    t_I = np.array(t_I, dtype=np.float32)

    pre_np   = [pv.cpu().numpy() for pv in precomputed_vectors]
    lambda_z = C1 * num_features

    delta_vars = []
    z_vars     = []
    range_exprs = []
    constraints = []

    for j in range(num_features):
        Mj = num_intervals_list[j] - 1
        dj = cp.Variable(Mj)
        delta_vars.append(dj)

        if   monotonic_info[j] == 'inc':
            constraints.append(dj >= 0)
            expr = cp.sum(dj)
        elif monotonic_info[j] == 'dec':
            constraints.append(dj <= 0)
            expr = -cp.sum(dj)
        else:
            expr = cp.norm1(dj)
        range_exprs.append(expr)

    p_terms = []
    for j in range(num_features):
        A_j = cp.Parameter(pre_np[j].shape, value=pre_np[j])
        p_j = A_j @ delta_vars[j]
        p_terms.append(p_j)

        z_j = cp.Variable(p_j.shape, nonneg=True)
        z_vars.append(z_j)
        constraints += [z_j >= p_j, z_j >= -p_j]

    p_expr       = sum(p_terms)
    abs_sum_expr = sum(z_vars)
    I_expr       = num_features - abs_sum_expr

    for expr in range_exprs:
        constraints.append(expr <= 1)

    data_loss = (
        cp.sum_squares(p_expr - t_p)
      + cp.sum_squares(I_expr - t_I)
      + lambda_z * cp.sum(abs_sum_expr)
    )

    prob = cp.Problem(cp.Minimize(data_loss), constraints)
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
            raise ValueError("Infeasible!")
        delta_opt.append(dj.value)
        diffs.append(np.max(np.abs(zj.value - np.abs(pj.value))))

    delta_tensors = [
        torch.tensor(arr, dtype=torch.float32, device=device)
        for arr in delta_opt
    ]
    return delta_tensors, diffs


def determine_relation(pij, Iij, rij, d):
    abs_p = abs(pij)
    if abs_p > d:
        if abs_p > rij:
            return 'P'  if pij > 0 else 'Undefined'
        else:
            return 'R'
    else:
        return 'I' if Iij > rij else 'R'

def generate_true_relations(labels):
    n = len(labels)
    rows = []
    for i in range(n):
        for j in range(i+1, n):
            diff = abs(labels[i] - labels[j])
            if diff > errorlimit:
                relation = 'P' if labels[i] > labels[j] else 'Undefined'
            else:
                relation = 'I'
            rows.append({'i': i, 'j': j, 'true_relation': relation})
    return pd.DataFrame(rows)

def evaluate_relations(
    feats, labels, delta_u_opt,
    num_intervals_list, max_diff_list,
    device, d
):
    feats_t = torch.tensor(feats, device=device)
    pre = precompute_piecewise_vectors(feats_t, num_intervals_list, max_diff_list, device)
    p_vec, marginals = scoring_function(pre, delta_u_opt)
    p = p_vec.cpu().numpy()
    marg_np = [m.cpu().numpy() for m in marginals]
    total_mag = sum(np.sum(np.abs(m)) for m in marg_np)

    target_total = len(marg_np) 
    scale_factor = target_total / (total_mag)

    marg_np_scaled = [m * scale_factor for m in marg_np]
    I = len(marg_np) - sum(np.abs(m) for m in marg_np_scaled)
    R = len(marg_np) - np.abs(p) - I

    _, pair_list = build_label_for_pairs(labels)
    pred_rows = []
    for idx, (i, j) in enumerate(pair_list):
        pred_rows.append({
            'i': i, 'j': j,
            'relation': determine_relation(p[idx], I[idx], R[idx], d)
        })
    df_pred = pd.DataFrame(pred_rows)
    df_true = generate_true_relations(labels)
    comp = df_pred.merge(df_true, on=['i','j'])
    comp['is_consistent'] = comp.apply(
        lambda r: r['relation']==r['true_relation'] or r['relation']=='R', axis=1
    )
    return comp['is_consistent'].mean()


def train_val_fold_once(
    feats_tr, labels_tr, feats_val, labels_val,
    num_intervals, C1, d,
    monotonic_info, max_diff_list, device
):
    num_features = feats_tr.shape[1]
    actual_num_intervals = [
        2 if len(np.unique(feats_tr[:, j]))==2 else num_intervals
        for j in range(num_features)
    ]
    pre_tr = precompute_piecewise_vectors(
        torch.tensor(feats_tr, device=device),
        actual_num_intervals, max_diff_list, device
    )
    y_pairs, _ = build_label_for_pairs(labels_tr)
    delta_u_opt, diffs = solve_qp_cvxpy(
        pre_tr, y_pairs,
        actual_num_intervals, monotonic_info,
        C1, float(num_features),
        device
    )
    perf = evaluate_relations(
        feats_val, labels_val,
        delta_u_opt, actual_num_intervals,
        max_diff_list, device, d
    )
    return perf, delta_u_opt, actual_num_intervals, diffs


def run_cv_search_and_test(
    features_np, labels_np, max_diff_list,
    interval_candidates=[3,5,7],
    C1_candidates=[0,1,2,3,4,5,10],
    d_candidates=[0.1, 0.2, 0.3],
  
    monotonic_info=None,
    n_splits=5,
    test_size=5,
    random_state=0,
    device=torch.device('cpu'),
    save_pkl_path="final_results.pkl"
):
    if monotonic_info is None:
        monotonic_info = ['inc'] * features_np.shape[1]

    idx = np.arange(len(labels_np))
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_size,
        random_state=random_state, shuffle=True
    )
    feats_tr, labs_tr = features_np[tr_idx], labels_np[tr_idx]
    feats_te, labs_te = features_np[te_idx], labels_np[te_idx]

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    param_grid = [
        (ni, c1, d)
        for ni in interval_candidates
        for c1 in C1_candidates
        for d in d_candidates
    
    ]

    def eval_params(ni, c1, d):
        scores = []
        for loc_tr, loc_val in skf.split(tr_idx):
            g_tr, g_val = tr_idx[loc_tr], tr_idx[loc_val]
            perf, _, _, _ = train_val_fold_once(
                features_np[g_tr], labels_np[g_tr],
                features_np[g_val], labels_np[g_val],
                ni, c1, d,
                monotonic_info, max_diff_list,
                device
            )
            scores.append(perf)
        return np.mean(scores)

    results = Parallel(n_jobs=-1)(
        delayed(eval_params)(ni, c1, d)
        for ni, c1, d in param_grid
    )
    best_idx = int(np.argmax(results))
    best_ni, best_c1, best_d = param_grid[best_idx]
    print(f"[CV] best=(ni={best_ni},C1={best_c1},d={best_d},perf={max(results):.4f}")

    _, delta_u_opt, actual_num_intervals, diffs = train_val_fold_once(
        feats_tr, labs_tr, feats_te, labs_te,
        best_ni, best_c1, best_d,
        monotonic_info, max_diff_list, device
    )
    final_perf = evaluate_relations(
        feats_te, labs_te,
        delta_u_opt, actual_num_intervals,
        max_diff_list, device, best_d
    )

    pre = precompute_piecewise_vectors(
        torch.tensor(feats_tr, device=device),
        actual_num_intervals, max_diff_list, device,
        features2=torch.tensor(feats_te, device=device)
    )
    p_vec, marginals = scoring_function(pre, delta_u_opt)
    p = p_vec.cpu().numpy()
    marg_np = [m.cpu().numpy() for m in marginals]
    I = len(marg_np) - sum(np.abs(m) for m in marg_np)
    R_arr = len(marg_np) - np.abs(p) - I
    _, pair_list = build_label_for_pairs(labs_te)
    R_count = sum(
        1 for idx in range(len(pair_list))
        if determine_relation(p[idx], I[idx], R_arr[idx], best_d) == 'R'
    )

    out = {
        'best_param': (best_ni, best_c1, best_d),
        'test_performance': final_perf,
        'R_count': R_count,
        'n_pairs': len(pair_list),
        'delta_scaled': delta_u_opt,
        'diffs': diffs
    }
    with open(save_pkl_path, "wb") as f:
        pickle.dump(out, f)
    return out


if __name__ == "__main__":
    df = pd.read_excel('D:/JORS/StructuralData.xlsx')
    features_np = df.iloc[:, [1,2,4]].to_numpy().astype(np.float32)
    labels_np   = df.iloc[:, 5].to_numpy().astype(np.float32)
    max_diff_list = [
        float(df.iloc[:, col].max() - df.iloc[:, col].min())
        for col in [1,2,4]
    ]
    monotonic_info = ['inc', 'dec', 'inc']

    perf_list, R_counts, best_params_list = [], [], []
    delta_scaled_list, diffs_list = [], []

    for i in range(100):
        res = run_cv_search_and_test(
            features_np, labels_np, max_diff_list,
            interval_candidates=[3,4,5,6,7,8,9],
            C1_candidates=list(range(0,8)),
            d_candidates=[0],
            monotonic_info=monotonic_info,
            n_splits=5,
            test_size=5,
            random_state=i+100,
            device=torch.device('cpu'),
            save_pkl_path=f"results_run_{i}.pkl"
        )
        perf_list.append(res['test_performance'])
        R_counts.append(res['R_count'])
        best_params_list.append(res['best_param'])
        delta_scaled_list.append(res['delta_scaled'])
        diffs_list.append(res['diffs'])
    pd.DataFrame({'performance': perf_list, 'R_count': R_counts, 'best_params': best_params_list, 'delta_scaled': delta_scaled_list, 'diffs': diffs_list}).to_csv('cv_results_summary.csv', index=False)

    perf_arr = np.array(perf_list)
    mean_perf = perf_arr.mean()
    STA = 1 - np.sqrt(((perf_arr - mean_perf) ** 2).sum() / len(perf_arr))
    mode_R = Counter(R_counts).most_common(1)[0][0]
    median_R = np.median(R_counts)
    q1, q3 = np.percentile(R_counts, [25, 75])
    n_pairs = len(build_label_for_pairs(labels_np)[1])
    
    print(f"100 runs: mean_perf={mean_perf:.4f}, STA={STA:.4f}, "
          f"mode_R={mode_R}, median_R={median_R}, IQR=[{q1},{q3}] out of {n_pairs} pairs")
    
    x = list(range(1, len(perf_list) + 1))
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    ax1.set_xlabel('Test index', fontsize=16)
    ax1.set_ylabel('Predictive performance', color='tab:blue', fontsize=18)
    ax1.plot(x, perf_list, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of incomparable pairs', color='tab:red', fontsize=18)
    ax2.plot(x, R_counts, 's-', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    
    ax1.grid(True)
    plt.show()

    best_run_idx = 1
    delta_scaled_best = delta_scaled_list[best_run_idx]
    diffs_best = diffs_list[best_run_idx]
    feature_names = ["TSS (mg/L)", "Salinity (PSU)", "Turbidity (NTU)"]
    feature_ranges = [
        (float(features_np[:, j].min()), float(features_np[:, j].max()))
        for j in range(len(feature_names))
    ]
    delta_list = [arr.cpu().numpy().flatten() for arr in delta_scaled_best]

    fig, axes = plt.subplots(1, len(feature_names), figsize=(6 * len(feature_names), 6))
    range_differences = []
    for idx, deltas in enumerate(delta_list):
        min_val, max_val = feature_ranges[idx]
        x_vals = np.linspace(min_val, max_val, len(deltas) + 1)
        y_vals = np.cumsum([0] + list(deltas))
        rd = np.sum(np.abs(deltas))
        range_differences.append(rd)
        ax = axes[idx]
        ax.plot(x_vals, y_vals, marker='o', linestyle='-')
        ax.set_title(f"{feature_names[idx]}\nRange: {rd:.5f}", fontsize=14)
        ax.set_xlabel("Differences", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Risk Preference", fontsize=12)
        ax.set_xticks(x_vals)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    print("Feature Weights:")
    weights = np.array(range_differences)
    weights /= weights.sum()
    for name, w in zip(feature_names, weights):
        print(f"{name}: {w:.5f}")