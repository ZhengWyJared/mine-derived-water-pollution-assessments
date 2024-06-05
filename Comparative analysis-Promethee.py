import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
from sklearn.model_selection import train_test_split, KFold
import time
from itertools import product
from scipy.optimize import minimize

def run_model(data, test_data):
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
            diff = abs(col4_i - col4_j)
    
            if diff > 0:
                if col4_i > col4_j:
                    Pij_given = 1
                    Iij_given = 0
                    Rij_given = 0
                else:
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
    num_rows = test_data.shape[0]
    combinations = list(product(range(1, num_rows + 1), range(1, num_rows + 1), range(1, num_cols + 1)))
    

    new_df = pd.DataFrame(combinations, columns=['i', 'j', 'k'])
    
 
    new_df = new_df.merge(results_df, on=['i', 'j', 'k'], how='left')
    
   
    new_df['p_ijk'] = new_df.apply(lambda row: row['pijk'] if row['sijk'] == 1 else 0, axis=1)
    new_df['p_jik'] = new_df.apply(lambda row: row['pijk'] if row['sijk'] == -1 else 0, axis=1)

    new_df['weighted_ijk'] = new_df['p_ijk'] * new_df['k'].apply(lambda x: w[x-1])
    new_df['weighted_jik'] = new_df['p_jik'] * new_df['k'].apply(lambda x: w[x-1])
    
    p_ij_df = new_df.groupby(['i', 'j']).agg({'weighted_ijk': 'sum', 'weighted_jik': 'sum'}).reset_index()
   
    final_df = p_ij_df.copy()

    i_less_than_j = final_df[final_df['i'] < final_df['j']]
    i_greater_than_j = i_less_than_j.copy()
    i_greater_than_j['i'], i_greater_than_j['j'] = i_less_than_j['j'], i_less_than_j['i']
    i_greater_than_j['weighted_ijk'], i_greater_than_j['weighted_jik'] = i_less_than_j['weighted_jik'], i_less_than_j['weighted_ijk']
    
 
    final_df = pd.concat([i_less_than_j, i_greater_than_j], ignore_index=True)
    
 
    final_df = final_df[final_df['i'] != final_df['j']]
    
  
    final_df.sort_values(by=['i', 'j'], inplace=True)
    

    final_df.reset_index(drop=True, inplace=True)
    
    Oi_plus = {}
    Oi_minus = {}
    

    n = max(final_df['i'].max(), final_df['j'].max())
    

    for i_value in range(1, n + 1):
      
        Oi_plus_sum = final_df[final_df['i'] == i_value]['weighted_ijk'].sum()
               
        Oi_minus_sum = final_df[final_df['j'] == i_value]['weighted_ijk'].sum()
        
 
        Oi_plus[i_value] = Oi_plus_sum / (n - 1)
        Oi_minus[i_value] = Oi_minus_sum / (n - 1)
    
   
    Oi_plus_df = pd.DataFrame(list(Oi_plus.items()), columns=['i', 'Oi_plus'])
    Oi_minus_df = pd.DataFrame(list(Oi_minus.items()), columns=['i', 'Oi_minus'])
    
    
    
    Oi_plus_dict = Oi_plus_df.set_index('i')['Oi_plus'].to_dict()
    Oi_minus_dict = Oi_minus_df.set_index('i')['Oi_minus'].to_dict()
    

    def determine_relation(i, j, Oi_plus_dict, Oi_minus_dict):
        Oi_plus = Oi_plus_dict.get(i, 0)
        Oj_plus = Oi_plus_dict.get(j, 0)
        Oi_minus = Oi_minus_dict.get(i, 0)
        Oj_minus = Oi_minus_dict.get(j, 0)
    
        if (Oi_plus > Oj_plus and Oi_minus < Oj_minus) or (Oi_plus == Oj_plus and Oi_minus < Oj_minus) or (Oi_plus > Oj_plus and Oi_minus == Oj_minus):
            return 'P'
        elif Oi_plus == Oj_plus and Oi_minus == Oj_minus:
            return 'I'
        elif (Oi_plus < Oj_plus and Oi_minus < Oj_minus) or (Oi_plus > Oj_plus and Oi_minus > Oj_minus):
            return 'R'
        return 'Undefined'
    
   
    if 'relation' not in final_df.columns:
        final_df['relation'] = final_df.apply(lambda row: determine_relation(row['i'], row['j'], Oi_plus_dict, Oi_minus_dict), axis=1)

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
    return performance_index, final_df[['i', 'j', 'relation']], result

data_path = r'DataAfterPreprocessing.xlsx'
data = pd.read_excel(data_path)

performance_indices = []
all_relations = []
results_x = []
results_success = []

start_time = time.time()

for i in range(100):
    train_data, test_data = train_test_split(data, test_size=5, random_state=i+3)
    
    performance_index, relations_df, result = run_model(train_data, test_data)
    
    performance_indices.append(performance_index)
    all_relations.append(relations_df)
    results_x.append(result.x)
    results_success.append(result.success)

end_time = time.time()
total_elapsed_time = end_time - start_time

average_performance = np.mean(performance_indices)
average_time = total_elapsed_time / 100



count_R = []

for df in all_relations:
    count = df['relation'].str.count('R').sum()
    count_R.append(count)

