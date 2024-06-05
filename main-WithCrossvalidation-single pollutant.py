import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
import time
from joblib import Parallel, delayed

def run_model(data, test_data, c):
    
    def calculate_pijk(m, n, a, b):
        if n - m <= a:
            return 0
        elif n - m > b:
            return 1
        else:
            return (n - m - a) / (b - a)

    def calculate_differences_and_pijk(data, a, b):
        num_rows = data.shape[0]
        num_cols = data.shape[1] - 4
        results = []

        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                for k in range(num_cols):
                    difference = data.iloc[i, k] - data.iloc[j, k]
                    m = min(data.iloc[i, k], data.iloc[j, k])
                    n = max(data.iloc[i, k], data.iloc[j, k])
                    sign = 1 if difference >= 0 else -1
                    pijk = calculate_pijk(m, n, a[k], b[k])
                    results.append({
                        'i': i + 1,
                        'j': j + 1,
                        'k': k + 1,
                        'pijk': pijk,
                        'sijk': sign
                    })

        results_df = pd.DataFrame(results)
        return results_df

    results = []
    n = data.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            col4_i = data.iloc[i, 4]
            col4_j = data.iloc[j, 4]

            if col4_i > col4_j:
                Pij_given = 1
                Iij_given = 0
                Rij_given = 0
            elif col4_i < col4_j:
                Pij_given = -1
                Iij_given = 0
                Rij_given = 0
            else:
                Pij_given = 0
                Iij_given = 1
                Rij_given = 0

            results.append({
                'i': i + 1,
                'j': j + 1,
                'Pij_given': Pij_given,
                'Iij_given': Iij_given,
                'Rij_given': Rij_given,
            })
    data_given_values = pd.DataFrame(results)

    def objective(params, data, data_given):
        num_cols = data.shape[1] - 4
        w = params[:num_cols]
        a = params[num_cols:2 * num_cols]
        b = params[2 * num_cols:3 * num_cols]

        results_df = calculate_differences_and_pijk(data, a, b)
        results_df['weighted_pijk'] = results_df['pijk'] * results_df['sijk'] * w[results_df['k'] - 1]
        results_df['weighted_1_minus_pijk'] = (1 - results_df['pijk']) * w[results_df['k'] - 1]

        grouped = results_df.groupby(['i', 'j'])
        pij = grouped['weighted_pijk'].sum()
        Iij = grouped['weighted_1_minus_pijk'].sum()
        rij = 1 - abs(pij) - Iij

        data_given_indexed = data_given.set_index(['i', 'j'])
        obj_value = 0
        for (i, j), p in pij.iteritems():
            given_row = data_given_indexed.loc[(i, j)]
            P = pij.loc[(i, j)]
            I = Iij.loc[(i, j)]
            r = rij.loc[(i, j)]
            obj_value += (P - given_row['Pij_given'])**2 + (I - given_row['Iij_given'])**2 + (r - given_row['Rij_given'])**2

        return obj_value

    num_cols = data.shape[1] - 4
    def constraints(num_cols):
        cons = []
        cons.append({'type': 'eq', 'fun': lambda x: np.sum(x[:num_cols]) - 1})
        for k in range(num_cols):
            cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[k]})
            cons.append({'type': 'ineq', 'fun': lambda x, k=k: 1 - x[k]})
        for k in range(num_cols):
            cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[2 * num_cols + k] - x[num_cols + k]})
            cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[num_cols + k]})
        return cons

    initial_guess = np.concatenate([np.ones(num_cols) / num_cols, np.zeros(num_cols), np.ones(num_cols)])
    
    result = minimize(objective, initial_guess, args=(data, data_given_values), method="SLSQP", constraints=constraints(num_cols))
    print("Optimized Results:")
    print(f"Variables (X): {result.x}")
    print(f"Objective Function Value: {result.fun}")
    print(f"Success: {result.success}")
    print(f"Number of Iterations: {result.nit}")

    w = result.x[:num_cols]
    a = result.x[num_cols:2 * num_cols]
    b = result.x[2 * num_cols:3 * num_cols]
    
    results_df = calculate_differences_and_pijk(test_data, a, b)
    results_df['weighted_pijk'] = results_df['pijk'] * results_df['sijk'] * w[results_df['k'] - 1]
    results_df['weighted_1_minus_pijk'] = (1 - results_df['pijk']) * w[results_df['k'] - 1]

    grouped = results_df.groupby(['i', 'j'])
    pij = grouped['weighted_pijk'].sum()
    Iij = grouped['weighted_1_minus_pijk'].sum()
    rij = 1 - abs(pij) - Iij
    final_df = pd.DataFrame({
        'pij': pij,
        'Iij': Iij,
        'rij': rij
    }).reset_index()

    def determine_relation(pij, Iij, rij):
        abs_pij = abs(pij)
        if abs_pij >= c:
            if abs_pij > rij:
                if pij < 0:
                    return 'Undefined'
                elif pij > 0:
                    return 'P'
            else:
                return 'R'
        else:
            if Iij > rij:
                return 'I'
            else:
                return 'R'

    final_df['relation'] = final_df.apply(
        lambda row: determine_relation(row['pij'], row['Iij'], row['rij']), axis=1
    )

    print("\nRelations between pairs (i, j):")
    print(final_df[['i', 'j', 'relation']].drop_duplicates().reset_index(drop=True))

    def generate_true_relations(test_data):
        num_rows = test_data.shape[0]
        true_relations = []
    
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                col4_i = test_data.iloc[i, 4]
                col4_j = test_data.iloc[j, 4]
                diff = abs(col4_i - col4_j)
    
                if diff > 0:
                    if col4_i > col4_j:
                        relation = 'P'
                    else:
                        relation = 'Undefined'
                else:
                    relation = 'I'
    
                true_relations.append({
                    'i': i + 1,
                    'j': j + 1,
                    'true_relation': relation
                })
    
        return pd.DataFrame(true_relations)

    true_relations_df = generate_true_relations(test_data)

    comparison_df = final_df.merge(true_relations_df, on=['i', 'j'])
    comparison_df['is_consistent'] = comparison_df.apply(
        lambda row: row['relation'] == row['true_relation'] or row['relation'] == 'R', axis=1
    )

    total_possible_relations = len(comparison_df)
    inconsistent_relations = len(comparison_df[~comparison_df['is_consistent']])
    performance_index = 1 - (inconsistent_relations / total_possible_relations)
    
 
    return performance_index, final_df[['i', 'j', 'relation']], result, true_relations_df

def cross_validate(train_data, c_values):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_c = None
    best_performance = -np.inf

    def parallel_run_model(train_data, val_data, c):
        performance_index, _, _, _ = run_model(train_data, val_data, c)
        return performance_index

    for c in c_values:
        performance_indices = Parallel(n_jobs=-1)(
            delayed(parallel_run_model)(train_data.iloc[train_index], train_data.iloc[val_index], c)
            for train_index, val_index in kf.split(train_data)
        )
        avg_performance = np.mean(performance_indices)
        print(f"c={c}, avg_performance={avg_performance}")  
        if (avg_performance > best_performance):
            best_performance = avg_performance
            best_c = c
    print(f"Best c={best_c}, Best Performance={best_performance}") 
    return best_c

c_values = np.linspace(0, 0.8, 10)
data_path = r'DataAfterPreprocessing.xlsx'
data = pd.read_excel(data_path)

performance_indices = []
all_relations = []
results_x = []
results_success = []
c_values_used = []
true_relations_list = [] 
predicted_relations_list = [] 

start_time = time.time()

for i in range(100):
    train_data, test_data = train_test_split(data, test_size=5, random_state=i+3)
    best_c = cross_validate(train_data, c_values)
    
    performance_index, relations_df, result, true_relations_df = run_model(train_data, test_data, best_c)
    
    c_values_used.append(best_c)
    
    performance_indices.append(performance_index)
    all_relations.append(relations_df)
    results_x.append(result.x)
    results_success.append(result.success)
    true_relations_list.append(true_relations_df) 
    predicted_relations_list.append(relations_df) 

end_time = time.time()
total_elapsed_time = end_time - start_time

average_performance = np.mean(performance_indices)
average_time = total_elapsed_time / 100



count_R = []

for df in all_relations:
    count = df['relation'].str.count('R').sum()
    count_R.append(count)
average_count_R = np.mean(count_R)
for index, count in enumerate(count_R):



import pickle


output_data = {
    'performance_indices': performance_indices,
    'all_relations': all_relations,
    'results_x': results_x,
    'results_success': results_success,
    'c_values_used': c_values_used,
    'true_relations_list': true_relations_list,
    'predicted_relations_list': predicted_relations_list,
    'average_performance': average_performance,
    'average_time': average_time,
    'total_elapsed_time': total_elapsed_time,
    'count_R': count_R
}

output_file_path = r'crossvalidationresults.pkl'
with open(output_file_path, 'wb') as f:
    pickle.dump(output_data, f)
L = len(performance_indices)
sum_squared_diff = sum((x - average_performance) ** 2 for x in performance_indices)

STA = 1 - np.sqrt(sum_squared_diff / L)
print(f'STA: {STA}')
