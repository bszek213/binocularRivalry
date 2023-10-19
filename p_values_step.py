import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Sample p-values DataFrame (replace this with your actual data)
p_values_df = pd.DataFrame({
    0: [np.nan, 0.001039, 0.000335, 0.000980, 0.000234, 0.008933, 0.011059, 0.001978, 0.000003, 0.012197, 0.012197],
    10: [0.001039, np.nan, 0.106549, 0.793127, 0.004684, 0.365596, 0.260472, 0.022320, 0.002130, 0.004889, 0.004889],
    20: [0.000335, 0.106549, np.nan, 0.160248, 0.011819, 0.681010, 0.429568, 0.000263, 0.006572, 0.000062, 0.000062],
    30: [0.000980, 0.793127, 0.160248, np.nan, 0.001050, 0.192295, 0.140824, 0.113315, 0.000264, 0.025448, 0.025448],
    40: [0.000234, 0.004684, 0.011819, 0.001050, np.nan, 0.004302, 0.027337, 0.002222, 0.034771, 0.001206, 0.001206],
    50: [0.008933, 0.365596, 0.681010, 0.192295, 0.004302, np.nan, 0.112484, 0.077366, 0.085627, 0.033523, 0.033523],
    60: [0.011059, 0.260472, 0.429568, 0.140824, 0.027337, 0.112484, np.nan, 0.068358, 0.359018, 0.034158, 0.034158],
    70: [0.001978, 0.022320, 0.000263, 0.113315, 0.002222, 0.077366, 0.068358, np.nan, 0.000278, 0.000110, 0.000110],
    80: [0.000003, 0.002130, 0.006572, 0.000264, 0.034771, 0.085627, 0.359018, 0.000278, np.nan, 0.000105, 0.000105],
    90: [0.012197, 0.004889, 0.000062, 0.025448, 0.001206, 0.033523, 0.034158, 0.000110, 0.000105, np.nan, np.nan],
    100: [0.012197, 0.004889, 0.000062, 0.025448, 0.001206, 0.033523, 0.034158, 0.000110, 0.000105, np.nan, np.nan]
})
p_values_df.index = p_values_df.columns

print(p_values_df)
# Number of comparisons
num_comparisons = 45  # Adjust the number of comparisons based on your data

# Calculate the Bonferroni corrected threshold
bonferroni_threshold = 0.05 / num_comparisons

# Create the heatmap with cell blocks
plt.figure(figsize=(10, 8))
ax = plt.gca()

for i in range(p_values_df.shape[0]):
    for j in range(p_values_df.shape[1]):
        p_value = p_values_df.iloc[i, j]
        if not np.isnan(p_value):
            if p_value < bonferroni_threshold:
                color = 'red'
            else:
                color = 'white'
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

# Overlay grid lines
plt.grid(True, color='black', linestyle='-', linewidth=1)

# Set x and y axis labels
plt.xticks(range(p_values_df.shape[1]), p_values_df.columns)
plt.yticks(range(p_values_df.shape[0]), p_values_df.index)

plt.ylabel('Percent Step Cycle (%)',fontweight='bold')
plt.xlabel('Percent Step Cycle (%)',fontweight='bold')
plt.savefig('alternations_step_cycle_t_test_cycle.png',dpi=400)