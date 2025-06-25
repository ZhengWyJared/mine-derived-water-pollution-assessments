import pandas as pd
import numpy as np
import torch
import cvxpy as cp
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from collections import Counter


def precompute_piecewise_vectors(
    features1, num_intervals_list, max_diff_list, device,
    features2=None
):
    if features2 is None:
        pair_indices = list(combinations(range(features1.shape[0]), 2))
    else:
        pair_indices = [(k, l)
                        for k in range(features1.shape[0])
                        for l in range(features2.shape[0])]

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
                    v[seg - 1] = ((abs_diff - intervals[seg - 1]) /
                                  (intervals[seg] - intervals[seg - 1]))
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
        for l in range(k + 1, n):
            diff = labels_1d[l] - labels_1d[k]
            if diff > errorlimit:
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
    C1, J, device=torch.device('cpu')
):
    num_features = len(num_intervals_list)
    

    t_p, t_I = [], []
    for val in labels_for_pairs:
        if val > 0:
            t_p.append(J); t_I.append(0.)
        elif val < 0:
            t_p.append(-J); t_I.append(0.)
        else:
            t_p.append(0.); t_I.append(J)
    t_p = np.array(t_p, dtype=np.float32)
    t_I = np.array(t_I, dtype=np.float32)

 
    lambda_z = C1 * num_features
    pre_np = [pv.cpu().numpy() for pv in precomputed_vectors]

    delta_vars, z_vars, range_exprs, constraints = [], [], [], []

    for j in range(num_features):
        Mj = num_intervals_list[j] - 1
        dj = cp.Variable(Mj)
        delta_vars.append(dj)

        if monotonic_info[j] == 'inc':
            constraints.append(dj >= 0)
            expr = cp.sum(dj)
        elif monotonic_info[j] == 'dec':
            constraints.append(dj <= 0)
            expr = -cp.sum(dj)
        else:
            expr = cp.norm1(dj) 
        range_exprs.append(expr)

    p_kl_j_exprs = []
    for j in range(num_features):
        A_j = cp.Parameter(pre_np[j].shape, value=pre_np[j])
        p_j = A_j @ delta_vars[j]
        p_kl_j_exprs.append(p_j)

        z_j = cp.Variable(p_j.shape, nonneg=True)
        z_vars.append(z_j)
        constraints += [z_j >= p_j, z_j >= -p_j]

    p_expr       = sum(p_kl_j_exprs)
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

    delta_u_opt = []
    diffs = []
    for j, (zj, pj) in enumerate(zip(z_vars, p_kl_j_exprs)):
        val_j = delta_vars[j].value
        if val_j is None:
            raise ValueError("Infeasible!")
        delta_u_opt.append(val_j)
        diffs.append(np.max(np.abs(zj.value - np.abs(pj.value))))

    delta_scaled = []
    for val_j in delta_u_opt:
        norm_j = np.linalg.norm(val_j)
        scale = 1.0 if norm_j > 1e-5 else 1e9
        delta_scaled.append(
            torch.tensor(val_j * scale, dtype=torch.float32, device=device))
    return delta_scaled, diffs

def compute_net_flows(p_vec, pair_list, n):

    preference_matrix = np.zeros((n, n))
    for idx, (i, j) in enumerate(pair_list):
        if p_vec[idx] > 0:
            preference_matrix[i, j] = p_vec[idx]
        elif p_vec[idx] < 0:
            preference_matrix[j, i] = -p_vec[idx]

    O_plus = np.sum(preference_matrix, axis=1) / (n - 1)
    O_minus = np.sum(preference_matrix, axis=0) / (n - 1)
    net_flows = O_plus - O_minus
    return O_plus, O_minus, net_flows

def determine_relation(i, j, O_plus, O_minus, net_flows, threshold=1e-5):
    Oi_plus, Oj_plus = O_plus[i], O_plus[j]
    Oi_minus, Oj_minus = O_minus[i], O_minus[j]

    if (abs(net_flows[i] - net_flows[j]) < threshold
        and abs(Oi_plus - Oj_plus) < threshold
        and abs(Oi_minus - Oj_minus) < threshold):
        return 'I'

    if ((Oi_plus > Oj_plus + threshold and Oi_minus < Oj_minus - threshold)
        or (Oi_plus >= Oj_plus - threshold and Oi_minus < Oj_minus - threshold)
        or (Oi_plus > Oj_plus + threshold and Oi_minus <= Oj_minus + threshold)):
        return 'P'

    if ((Oi_plus < Oj_plus - threshold and Oi_minus > Oj_minus + threshold)
        or (Oi_plus <= Oj_plus + threshold and Oi_minus > Oj_minus + threshold)
        or (Oi_plus < Oj_plus - threshold and Oi_minus >= Oj_minus - threshold)):
        return 'Undefined'

    return 'R'

def generate_true_relations(labels):
    n = len(labels)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
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
    device, d=None
):

    feats_t = torch.tensor(feats, device=device)
    pre = precompute_piecewise_vectors(
        feats_t, num_intervals_list, max_diff_list, device)

    p_vec, _ = scoring_function(pre, delta_u_opt)
    p = p_vec.cpu().numpy()

    _, pair_list = build_label_for_pairs(labels)
    n = len(labels)

    O_plus, O_minus, net_flows = compute_net_flows(p, pair_list, n)

    pred_rows = []
    for i, j in pair_list:
        pred_rows.append({
            'i': i,
            'j': j,
            'relation': determine_relation(
                i, j, O_plus, O_minus, net_flows)
        })
    df_pred = pd.DataFrame(pred_rows)

    df_true = generate_true_relations(labels)
    comp = df_pred.merge(df_true, on=['i', 'j'])
    comp['is_consistent'] = comp.apply(
        lambda r: r['relation'] == r['true_relation'] or r['relation'] == 'R',
        axis=1
    )
    return comp['is_consistent'].mean()


def train_val_fold_once(
    feats_tr, labels_tr, feats_val, labels_val,
    num_intervals, C1, d,
    monotonic_info, max_diff_list, device
):
    num_features = feats_tr.shape[1]
    actual_num_intervals = [
        2 if len(np.unique(feats_tr[:, j])) == 2 else num_intervals
        for j in range(num_features)
    ]
    pre_tr, _ = precompute_piecewise_vectors(
        torch.tensor(feats_tr, device=device),
        actual_num_intervals, max_diff_list, device), None
    y_pairs, _ = build_label_for_pairs(labels_tr)
    delta_u_opt, _ = solve_qp_cvxpy(
        pre_tr, y_pairs,
        actual_num_intervals, monotonic_info,
        C1, float(num_features), device
    )
    perf = evaluate_relations(
        feats_val, labels_val,
        delta_u_opt, actual_num_intervals,
        max_diff_list, device, d
    )
    return perf, delta_u_opt, actual_num_intervals

def run_cv_search_and_test(
    features_np, labels_np, max_diff_list,
    interval_candidates=[3, 5, 7],
    C1_candidates=[0.01, 0.1],
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
            perf, _, _ = train_val_fold_once(
                features_np[g_tr], labels_np[g_tr],
                features_np[g_val], labels_np[g_val],
                ni, c1, d,
                monotonic_info, max_diff_list,
                device
            )
            scores.append(perf)
        return np.mean(scores)

    results = Parallel(n_jobs=-1)(
        delayed(eval_params)(ni, c1, d) for ni, c1, d in param_grid
    )
    best_ni, best_c1, best_d = param_grid[int(np.argmax(results))]
    print(f"[CV] best=(ni={best_ni},C1={best_c1},d={best_d}), perf={max(results):.4f}")

    _, delta_u_opt, actual_num_intervals = train_val_fold_once(
        feats_tr, labs_tr, feats_te, labs_te,
        best_ni, best_c1, best_d,
        monotonic_info, max_diff_list,
        device
    )
    final_perf = evaluate_relations(
        feats_te, labs_te,
        delta_u_opt, actual_num_intervals,
        max_diff_list, device, best_d
    )

    _, pair_list = build_label_for_pairs(labs_te)
    pre_test = precompute_piecewise_vectors(
        torch.tensor(feats_tr, device=device), actual_num_intervals,
        max_diff_list, device,
        features2=torch.tensor(feats_te, device=device)
    )
    p_vec_test, _ = scoring_function(pre_test, delta_u_opt)
    p_test = p_vec_test.cpu().numpy()
    O_plus_test, O_minus_test, net_flows_test = compute_net_flows(
        p_test, pair_list, len(labs_te)
    )
    R_count = sum(
        1 for (i, j) in pair_list
        if determine_relation(i, j, O_plus_test, O_minus_test, net_flows_test) == 'R'
    )

    return {
        'best_param': (best_ni, best_c1, best_d),
        'test_performance': final_perf,
        'R_count': R_count,
        'n_pairs': len(pair_list)
    }


if __name__ == "__main__":
    df = pd.read_excel('D:/JORS/StructuralData.xlsx')
    features_np = df.iloc[:, [1, 2, 4]].to_numpy().astype(np.float32)
    labels_np = df.iloc[:, 5].to_numpy().astype(np.float32)
    max_diff_list = [
        float(df.iloc[:, col].max() - df.iloc[:, col].min())
        for col in [1, 2, 4]
    ]
    monotonic_info = ['inc', 'dec', 'inc']

    perf_list, R_counts = [], []
    for i in range(100):
        res = run_cv_search_and_test(
            features_np, labels_np, max_diff_list,
            interval_candidates=[2, 3, 4, 5, 6],
            C1_candidates=[0, 0.1, 0.5, 1, 5, 10, 12, 15],
            d_candidates=[0],
            monotonic_info=monotonic_info,
            n_splits=5, test_size=5,
            random_state=i + 100,
            device=torch.device('cpu')
        )
        perf_list.append(res['test_performance'])
        R_counts.append(res['R_count'])

    perf_arr = np.array(perf_list)
    mean_perf = perf_arr.mean()
    STA = 1 - np.sqrt(((perf_arr - mean_perf)**2).sum() / len(perf_arr))
    mode_R = Counter(R_counts).most_common(1)[0][0]
    median_R = np.median(R_counts)
    q1, q3 = np.percentile(R_counts, [25, 75])
    n_pairs = res['n_pairs']

    print(
        f"100 runs: mean_perf={mean_perf:.4f}, STA={STA:.4f}, "
        f"mode_R={mode_R}, median_R={median_R}, IQR=[{q1},{q3}] out of {n_pairs} pairs"
    )    

    x  = list(range(1, len(perf_list) + 1))
    y1 = perf_list
    y2 = R_counts
    

    fig, ax1 = plt.subplots(figsize=(12, 8))
    

    color1 = 'tab:blue'
    ax1.set_xlabel('Test index', fontsize=16)
    ax1.set_ylabel('Predictive performance', color=color1, fontsize=18)
    ax1.plot(x, y1, 'o-', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Number of incomparable pairs', color=color2, fontsize=18)
    ax2.plot(x, y2, 's-', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

    ax1.grid(True)
    plt.show()

