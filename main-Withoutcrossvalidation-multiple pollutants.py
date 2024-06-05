import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
import time
from joblib import Parallel, delayed

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
        num_cols = data.shape[1] - 8
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
            a = b = c = 0
            for k in [3, 4, 5, 6]:  
                col_i = data.iloc[i, k]
                col_j = data.iloc[j, k]
                diff = abs(col_i - col_j)
    
                if diff > 0:
                    if col_i > col_j:
                        a += 1
                    else:
                        b += 1
                else:
                    c += 1
            
            if a + c > 0:
                if a > b:
                    Pij_given = (a - b) / (a + c)
                    Iij_given = c / (a + c)
                    Rij_given = b / (a + c)
                else:
                    Pij_given = (a - b) / (b + c)
                    Iij_given = c / (b + c)
                    Rij_given = a / (b + c)
            else:
                Pij_given = 0
                Iij_given = 0
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
            obj_value += (P - given_row['Pij_given'])**2 + (I - given_row['Iij_given'])**2+(r - given_row['Rij_given'])**2

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
        if abs_pij > 0:
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
                a = b = c = 0
            for k in [3, 4, 5, 6]:  # The correct indices for 4th, 5th, and 6th columns
                col_i = test_data.iloc[i, k]
                col_j = test_data.iloc[j, k]
                diff = abs(col_i - col_j)
    
                if diff > 0:
                    if col_i > col_j:
                        a += 1
                    else:
                        b += 1
                else:
                    c += 1
            if a + c > 0:
                if a > b:
                    Pij_test = (a - b) / (a + c)
                    Iij_test = c / (a + c)
                    Rij_test = b / (a + c)
                else:
                    Pij_test = (a - b) / (b + c)
                    Iij_test = c / (b + c)
                    Rij_test = a / (b + c)
            else:
                Pij_test = 0
                Iij_test = 0
                Rij_test = 0
           
    
            relation = determine_relation(Pij_test, Iij_test, Rij_test)
            
            true_relations.append({
                'i': i + 1,
                'j': j + 1,
                'Pij_test': Pij_test,
                'Iij_test': Iij_test,
                'Rij_test': Rij_test,
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
    
    # Return true_relations_df and final_df as additional outputs
    return performance_index, final_df[['i', 'j', 'relation']], result, true_relations_df




data_path = r'DataAfterPreprocessing.xlsx'
data = pd.read_excel(data_path)

performance_indices = []
all_relations = []
results_x = []
results_success = []
c_values_used = []
true_relations_list = []  # List to store true relations DataFrames
predicted_relations_list = []  # List to store predicted relations DataFrames

start_time = time.time()

for i in range(100):
    train_data, test_data = train_test_split(data, test_size=5, random_state=i+3)
    
    
    performance_index, relations_df, result, true_relations_df = run_model(train_data, test_data)
    
   
    
    performance_indices.append(performance_index)
    all_relations.append(relations_df)
    results_x.append(result.x)
    results_success.append(result.success)
    true_relations_list.append(true_relations_df)  # Save true relations DataFrame
    predicted_relations_list.append(relations_df)  # Save predicted relations DataFrame

end_time = time.time()
total_elapsed_time = end_time - start_time

average_performance = np.mean(performance_indices)
average_time = total_elapsed_time / 100



count_R = []

for df in all_relations:
    count = df['relation'].str.count('R').sum()
    count_R.append(count)
average_count_R = np.mean(count_R)

