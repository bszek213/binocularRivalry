import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Function to display images at specific coordinates
def add_image(image_path, ax, x, y, zoom=0.1):
    image = plt.imread(image_path)
    imagebox = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)   

def add_text_label(label, x, y, ax, fontsize=12):
    ax.text(x, y, label, fontsize=fontsize, ha='center', va='top')

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
plt.figure(figsize=(12, 8))
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

# Overlay text labels below the x-axis
# x_intervals = [1, 8, 10]  # Adjust as needed (0 to 100)
# y = -1  # Adjust the y-coordinate to place labels below the x-axis
# labels = ["Heel Strike", "Midstance", "Toe Off"]

x_intervals = [0.5, 4, 7, 9.5]  # Adjust as needed (0 to 100)
y = -0.5  # Slightly above the x-axis label
labels = ["Toe Off", "Heel Strike", "Midstance", "Toe Off"]

for x, label in zip(x_intervals, labels):
    ax.text(x, y, label, fontsize=12, ha='center', va='top', fontweight='bold')

# Add labels along the y-axis
y_intervals = [0.5, 4, 7, 9.5]  # Adjust as needed (0 to 100)
x_val = -0.5  # Slightly above the x-axis label
# labels_y = ["Label1", "Label2", "Label3"]

for y, label in zip(y_intervals, labels):
    ax.text(x_val, y, label, fontsize=12, ha='right', va='center', fontweight='bold', rotation=90)

# for x, label in zip(x_intervals, labels):
#     add_text_label(label, x, y, ax, fontsize=12)
# Overlay images below the x-axis
# x_intervals = [.1]  # Adjust as needed (0 to 10)
# y = 0.1  # Adjust the y-coordinate to place images below the x-axis
# image_paths = [
#     "gait_images/midswing.png",
#     # "gait_images/heel_strike1.png",
#     # "gait_images/start_midstance.png"
# ]
# for y in x_intervals:
#     add_image(image_paths.pop(0), ax_images, x, y, zoom=0.1)
# for x in x_intervals:
#     add_image(image_paths.pop(0), x, y)

# Overlay images at specified intervals
# x_intervals = [10, 30, 60]  # Adjust as needed (0 to 100)
# y_intervals = [20, 50, 80]  # Adjust as needed (0 to 100)
# image_paths = ["gait_images/midswing.png",
#                 "gait_images/heel_strike1.png","gait_images/start_midstance.png",
#                 "gait_images/midstance.png", "gait_images/heel_strike_opp_foot.png",
#                 "gait_images/midswing.png"
#                 ]
# for x, y in zip(x_intervals,y_intervals):
#         add_image(image_paths.pop(0), x, y, zoom=0.1)

#GAIT IMAGES
# Specify the paths of your images
# image_paths = ["midswing.png",
#                 "heel_strike1.png","start_midstance.png",
#                 "midstance.png", "heel_strike_opp_foot.png",
#                 "midswing.png"]
# # Specify the x-axis positions for each image
# x_positions = [0,
#                 17, 42,
#                 60,74,
#                 94]
# for path, x in zip(image_paths, x_positions):
#     # Load and plot each image at the specified x-axis position
#     image = plt.imread(path)
#     if path == "midswing.png" or path == "midstance.png" or path == "midswing1.png":
#         ax[3].imshow(image, extent=[x, x+6, 0, 1],aspect="auto") 
#     else:
#         ax[3].imshow(image, extent=[x, x+10, 0, 1],aspect="auto") 
# ax.tick_params(axis='both', which='both', labelsize=12, pad=5, fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylabel('Percent Step Cycle (%)',fontweight='bold',fontsize=14,labelpad=25)
plt.xlabel('Percent Step Cycle (%)',fontweight='bold',fontsize=14, labelpad=25)
plt.savefig('alternations_step_cycle_t_test_cycle.png',dpi=400)