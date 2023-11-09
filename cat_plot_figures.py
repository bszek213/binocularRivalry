import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mean_switch_stand = [0.825, 0.7208333333333333, 0.7458333333333333, 0.3375, 0.65, 0.9041666666666667]
mean_switch_walk = [0.8833333333333333, 0.68125, 0.6104166666666667, 0.35833333333333334, 0.6729166666666667, 0.975]

# # Define the x-axis categories
# categories = ["Stand", "Walk"]

# # Create a list of positions for the categories on the x-axis
# x_positions = np.arange(len(categories))

# # Create a single plot with individual traces
# for x, y in zip(mean_switch_stand, mean_switch_walk):
#     plt.scatter([x_positions[0], x_positions[1]], [x, y], color='blue', marker='o')
#     plt.plot([x_positions[0], x_positions[1]], [x, y], color='blue', linestyle='-', linewidth=1)

# # Set the x-axis ticks and labels
# plt.xticks(x_positions, categories)
# plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2) #figsize=(10, 10)
categories = ["Stand", "Walk"]
x_positions = np.arange(len(categories))
#Alternation rates

# ax[0,0].bar('Stand',np.mean(mean_switch_stand),
#         yerr=np.std(mean_switch_stand),
#         color='tab:blue', capsize=5, label='Stand',alpha=0.5)
# for x, y in zip(mean_switch_stand, mean_switch_walk):
#     ax[0,0].scatter([x_positions[0], x_positions[1]], [x, y], color='grey', marker='o',alpha=0.5)
#     ax[0,0].plot([x_positions[0], x_positions[1]], [x, y], color='grey', linestyle='-', linewidth=1,alpha=0.5)
# ax[0,0].set_xticks(x_positions, categories)
sns.catplot(ax=ax[0,0],data=pd.DataFrame({"Stand":mean_switch_stand, "Walk":mean_switch_walk}),
            kind="point", color='blue',capsize=.1,ci=95,join=True, markers='o',scale=0.5)
ax[0,0].set_ylabel('Alternation Rate (Hz)')
plt.show()