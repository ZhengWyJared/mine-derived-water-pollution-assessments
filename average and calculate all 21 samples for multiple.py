import numpy as np
import pandas as pd




results_array = np.array(results_x)


num_cols = 3


w_values = results_array[:, :num_cols]
a_values = results_array[:, num_cols:2 * num_cols]
b_values = results_array[:, 2 * num_cols:3 * num_cols]

# 计算w, a, b的平均值
w_mean = np.mean(w_values, axis=0)
a_mean = np.mean(a_values, axis=0)
b_mean = np.mean(b_values, axis=0)
data_path = r'DataAfterPreprocessing.xlsx'
data = pd.read_excel(data_path)

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

results_df = calculate_differences_and_pijk(data, a_mean, b_mean)
results_df['weighted_pijk'] = results_df['pijk'] * results_df['sijk'] * w_mean[results_df['k'] - 1]
results_df['weighted_1_minus_pijk'] = (1 - results_df['pijk']) * w_mean[results_df['k'] - 1]
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

def generate_true_relations(data):
        num_rows = data.shape[0]
        true_relations = []
    
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                a = b = c = 0
            for k in [3, 4, 5, 6, 7]:  # The correct indices for 4th, 5th, and 6th columns
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
true_relations_df = generate_true_relations(data)

comparison_df = final_df.merge(true_relations_df, on=['i', 'j'])
comparison_df['is_consistent'] = comparison_df.apply(
    lambda row: row['relation'] == row['true_relation'] or row['relation'] == 'R', axis=1
)

total_possible_relations = len(comparison_df)
inconsistent_relations = len(comparison_df[~comparison_df['is_consistent']])
performance_index = 1 - (inconsistent_relations / total_possible_relations)

count_R_ALL = comparison_df['relation'].str.count('R').sum()

print(f"Performance Index: {performance_index}")
print(f"Total 'R' relations: {count_R_ALL}")
