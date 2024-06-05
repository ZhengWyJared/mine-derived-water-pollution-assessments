import pandas as pd


df = pd.DataFrame({'count_R': count_R})
df.to_csv('count_R.csv', index=False)


df2 = pd.DataFrame({'performance_indices': performance_indices})
df2.to_csv('performance_indices.csv', index=False)


import matplotlib.pyplot as plt


data_performance = pd.read_csv('performance_indices.csv')
data_count = pd.read_csv('count_R.csv')


x = range(1, 101)  
y1 = data_performance['performance_indices']  
y2 = data_count['count_R'] / 1  


fig, ax1 = plt.subplots(figsize=(12, 8))


color = 'tab:blue'
ax1.set_xlabel('Number of tests', fontsize=14) 
ax1.set_ylabel('Predictive performance', color=color, fontsize=14)  
ax1.plot(x, y1, 'o-', color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Number of the incomparable pairs', color=color, fontsize=14)  
ax2.plot(x, y2, 's-', color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)


ax1.grid(True)
plt.show()



df3 = pd.DataFrame({'c_values_used': c_values_used})
df3.to_csv('c_values_used.csv', index=False)


data_c_values = pd.read_csv('c_values_used.csv')


x = range(1, len(data_c_values) + 1) 
y = data_c_values['c_values_used']


plt.figure(figsize=(12, 8))
plt.plot(x, y, marker='o', linestyle='-', color='b')


plt.xlabel('Number of tests', fontsize=14)
plt.ylabel('The value of α in each test', fontsize=14)

plt.grid(True)


plt.show()

