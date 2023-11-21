import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks,find_peaks_cwt, savgol_filter
import os
import glob
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, norm, uniform, ttest_rel 
import statsmodels.api as sm
from fitter import Fitter, get_common_distributions, get_distributions
from tqdm import tqdm
from matplotlib import rc,rcParams
import matplotlib.patches as patches
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

def read_all_files():
    directory = "/home/brianszekely/Desktop/ProjectsResearch/Binocular_Rivalry/Data/*/" #POPOS
    csv_files_walk = [] 
    csv_files_track = []
    for abs_path in glob.glob(directory, recursive = True):
        temp_walk = glob.glob(os.path.join(abs_path,'Walking',"*.csv"))
        temp_track = glob.glob(os.path.join(abs_path,'Tracking',"*.csv"))
        if 'Response' in str(temp_walk):
            for file in temp_walk:
                if 'Response' in str(file):
                    csv_files_walk.append(file)
        #extract tracking
        for file in temp_walk:
                if 'Tracking' in str(file):
                    csv_files_track.append(file)
    csv_files_walk = sorted(csv_files_walk)
    csv_files_track = sorted(csv_files_track)

    return csv_files_walk, csv_files_track

def read_stand_files():
    directory = "/home/brianszekely/Desktop/ProjectsResearch/Binocular_Rivalry/Data/*/" #POPOS
    csv_files = [] 
    for abs_path in glob.glob(directory, recursive = True):
        temp_walk = glob.glob(os.path.join(abs_path,'Stationary',"*.csv"))
        if 'Response' in str(temp_walk):
            for file in temp_walk:
                if 'Response' in str(file):
                    csv_files.append(file)

    return sorted(csv_files)

def alt_rate_stand(df):
    df['greenButtonPressed'][df['greenButtonPressed'] == 1] = -1
    # Initialize variables to keep track of trigger states and time
    green_state = df['greenButtonPressed']
    red_state = df['redButtonPressed']
    time = df['t']

    #GET TOTAL TIME MIXED PERCEPTS
    time_diff = df['t'].diff()
    mixed_time = 0
    for t,green,red in zip(time_diff,green_state,red_state):
        if green == 0 and red == 0 and not np.isnan(t):
            mixed_time += t
    percent_total_mixed = (mixed_time / df['t'].iloc[-1]) * 100

    with open('mixed_percentage_per_trial_stand.txt', 'a') as file:
        file.write(f'{percent_total_mixed}\n')

def alt_rate_try(df,time_moments):
    """
    TODO: FIX the alternations to be set back to the location of when they switch from
    0 to 1 or 0 to -1
    """
    df['greenButtonPressed'][df['greenButtonPressed'] == 1] = -1
    # Initialize variables to keep track of trigger states and time
    green_state = df['greenButtonPressed']
    red_state = df['redButtonPressed']
    time = df['t']

    #GET TOTAL TIME MIXED PERCEPTS
    time_diff = df['t'].diff()
    mixed_time = 0
    for t,green,red in zip(time_diff,green_state,red_state):
        if green == 0 and red == 0 and not np.isnan(t):
            mixed_time += t
    percent_total_mixed = (mixed_time / df['t'].iloc[-1]) * 100

    with open('mixed_percentage_per_trial.txt', 'a') as file:
        file.write(f'{percent_total_mixed}\n')
    # Define the time constraint (500 milliseconds)
    time_constraint = 0.3  # seconds

    # Create a list to store the times of alternations
    alternation_times_green_list = []
    alternation_times_red_list = []
    alternation_times_red = 0
    alternation_times_green = 0


    # Iterate through the DataFrame rows
    # print(df)
    # print(df.index[0])
    # print(df.index[-1])
    # print(range(df.index[0]+1, df.index[-1]))
    # input()
    for i in range(df.index[0]+1, df.index[-1]):
        prev_green_state = green_state[i-1]
        prev_red_state = red_state[i-1]
        current_green_state = green_state[i]
        current_red_state = red_state[i]

        #GREEN ALTERNATIONS
        if ((prev_green_state == 0 and current_green_state == -1)):
            start_switch_green = time[i]
            increment = i
            end_switch_green = None
            while True:
                    if (red_state[increment] == 0 and red_state[increment-1] == 1):
                        end_switch_green = time[increment]
                        break
                    if (start_switch_green - time[increment]) > time_constraint:
                        break
                    increment += 1
                    if increment >= df.index[-1]:
                        break
            if end_switch_green != None:
                if (end_switch_green - start_switch_green) < time_constraint:
                    alternation_times_green += 1
                    alternation_times_green_list.append(increment)
            # i = increment

        #edge case green switch
        if ((green_state[i-1] == 0 and green_state[i] == -1)
              and (red_state[i-2] == 1 and red_state[i-1] == 0)):
            alternation_times_green += 1
            alternation_times_green_list.append(i)

        #edge case red goes from 1 to 0 and then the green happens later
        if (current_red_state == 0 and prev_red_state == 1):
            time_start_edge = time[i]
            increment = i
            while True:
                if (green_state[increment] == -1 and green_state[increment-1] == 0):
                    if (time[increment] - time_start_edge) < time_constraint:
                        alternation_times_green += 1
                        alternation_times_green_list.append(increment)
                    elif (time[increment] - time_start_edge) > time_constraint:
                        break
                increment += 1
                if increment >= df.index[-1]:
                    break

        #RED ALTERNATIONS
        if ((prev_red_state == 0 and current_red_state == 1)):
            start_switch_red = time[i]
            end_switch_red = None
            increment = i
            while True:
                    if (green_state[increment] == 0 and green_state[increment-1] == -1):
                        end_switch_red = time[increment]
                        break
                    if (start_switch_red - time[increment]) > time_constraint:
                        break
                    increment += 1
                    if increment >= df.index[-1]:
                        break
            if end_switch_red != None:
                if (end_switch_red - start_switch_red) < time_constraint:
                    alternation_times_red += 1
                    alternation_times_red_list.append(increment)
            # i = increment
        #edge case red switch
        if i < df.index[-1]:
            if ((red_state[i-1] == 0 and red_state[i] == 1)
                and (green_state[i-2] == -1 and green_state[i-1] == 0)):
                alternation_times_red += 1
                alternation_times_red_list.append(i)

        #edge case green goes from -1 to 0 and then the red happens later
        if (current_green_state == 0 and prev_green_state == -1):
            time_start_edge = time[i]
            increment = i
            while True:
                if (red_state[increment] == 1 and red_state[increment-1] == 0):
                    if (time[increment] - time_start_edge) < time_constraint:
                        alternation_times_red += 1
                        alternation_times_red_list.append(increment)
                    elif (time[increment] - time_start_edge) > time_constraint:
                        break
                increment += 1
                if increment >= df.index[-1]:
                    break

    # # Plot the data
    # plt.plot(time, green_state, marker='*', markersize=4, label='Green Button')
    # plt.plot(time, red_state, marker='*', markersize=4, label='Red Button')
    # # print(time[alternation_times_green])
    # # Mark alternation points on the plot
    # plt.scatter(time[alternation_times_green_list], 
    #             green_state[alternation_times_green_list], color='green', marker='o',s=50, label='Alternation Point')
    # plt.scatter(time[alternation_times_red_list], 
    #             red_state[alternation_times_red_list], color='red', marker='o',s=50, label='Alternation Point')
    # # Add labels and legend
    # plt.xlabel('Time (s)')
    # plt.ylabel('Button State')
    # plt.legend()
    # # Show the plot
    # plt.show()


    #extract stride moments
    global_switches = sorted(list(set(alternation_times_green_list + alternation_times_red_list)))
    save_alternation_stride_cycle = []
    # print(time)
    # print(time.index[-1])
    for alt_switch_time in time[global_switches]:
        # print(f'alternation: {alt_switch_time}')
        # print(np.abs(time_moments.values - alt_switch_time))
        # Calculate the absolute differences between alt_switch_time and value2
        closest_index = np.argmin(np.abs(time_moments - alt_switch_time))
        #IF the time is less than
        if time_moments.iloc[closest_index] > alt_switch_time:
            start_time = time_moments.iloc[closest_index-1]
            end_time = time_moments.iloc[closest_index]
            percentage = (alt_switch_time - start_time) / (end_time - start_time) * 100
            save_alternation_stride_cycle.append(percentage)
            # print(f'percentage LESS: {percentage}')
        elif time_moments.iloc[closest_index] < alt_switch_time:
            start_time = time_moments.iloc[closest_index]
            if (closest_index+1) >= len(time_moments):
                end_time = time_moments.iloc[closest_index]
            else:
                # print(closest_index)
                # print(time.index[-1])
                end_time = time_moments.iloc[closest_index+1]
            percentage = (alt_switch_time - start_time) / (end_time - start_time) * 100
            save_alternation_stride_cycle.append(percentage)
            # print(f'percentage GREATER: {percentage}')
    return save_alternation_stride_cycle


def calc_alternation_location(resp_data,time_moments):
    switch_data = resp_data[['t','greenButtonPressed','redButtonPressed']]
    switch_data['greenButtonPressed'][switch_data['greenButtonPressed'] == 1] = -1
    global_switches = []
    green_switch = []
    red_switch = []
    curr_trigger = 'black'
    closest_green = None
    closest_red = None
    try:
        for i in range(len(switch_data)):
            if red_switch and green_switch:
                closest_red = i - red_switch[-1]
                closest_green = i - green_switch[-1]
                if closest_red < closest_green:
                    curr_trigger = "Green"
                else:
                    curr_trigger = "Red"
            else:
                curr_trigger = "black"
            #Red responses
            if ((switch_data['redButtonPressed'].iloc[i-1] == 0)
                and (switch_data['redButtonPressed'].iloc[i] == 1)):
                if ((switch_data['greenButtonPressed'].iloc[i] == 0)):
                    if curr_trigger == "Red" or curr_trigger == "black":
                        red_switch.append(i)
                        global_switches.append(i)
    #                 curr_trigger = "Green"
                else:
                    check = i
                    check_start = i
                    while True:
                        if ((switch_data['greenButtonPressed'].iloc[check-1] == -1)
                        and (switch_data['greenButtonPressed'].iloc[check] == 0)):
                            break
                        else:
                            check +=1
                        if (switch_data['redButtonPressed'].iloc[check] == 0):
    #                         curr_trigger = " "
                            break
                        if check == len(switch_data):
                            break
    #                 if i > 9200 and i < 9500:
    #                     print('curr trigger',curr_trigger)
                    if (check - check_start) < 300:
                        if red_switch and closest_green:
                            closest_red = i - red_switch[-1]
                            closest_green = i - green_switch[-1]
                            if closest_green < closest_red:
                                if curr_trigger == "Red" or curr_trigger == "black":
                                    red_switch.append(i)
                                    global_switches.append(i)
    #                                 curr_trigger = "Green"
                        else:
                            if curr_trigger == "Red" or curr_trigger == "black":
                                    red_switch.append(i)
                                    global_switches.append(i)
    #                                 curr_trigger = "Green"
            #Green resp
            if ((switch_data['greenButtonPressed'].iloc[i-1] == 0)
                and (switch_data['greenButtonPressed'].iloc[i] == -1)):
                if ((switch_data['redButtonPressed'].iloc[i] == 0)):
                    if curr_trigger == "Green" or curr_trigger == "black":
                        green_switch.append(i)
                        global_switches.append(i)
    #                 curr_trigger = "Red"
                else:
                    check = i
                    check_start = i
                    while True:
                        if ((switch_data['redButtonPressed'].iloc[check-1] == 1)
                        and (switch_data['redButtonPressed'].iloc[check] == 0)):
                            break
                        else:
                            check += 1
                        if (switch_data['greenButtonPressed'].iloc[check] == 0):
    #                         curr_trigger = " "
                            break
                        if check == len(switch_data):
                            break
    #                 if i > 9200 and i < 9500:
    #                     print('curr trigger',curr_trigger)
                    if (check - check_start) < 300:
                            if closest_green is not None and closest_red is not None:
                                if closest_green > closest_red:
                                    if curr_trigger == "Green" or curr_trigger == "black":
                                        green_switch.append(i)
                                        global_switches.append(i)
    except:
        print('nothing just want to see')
    #Find where in step cycle the button was pressed
    save_alternation_stride_cycle = []
    for alt_switch_time in switch_data['t'].iloc[global_switches]:
        print(f'alternation: {alt_switch_time}')
        # Calculate the absolute differences between alt_switch_time and value2
        closest_index = np.argmin(np.abs(time_moments - alt_switch_time))
        #IF the time is less than
        if time_moments.iloc[closest_index] > alt_switch_time:
            start_time = time_moments.iloc[closest_index-1]
            end_time = time_moments.iloc[closest_index]
            percentage = (alt_switch_time - start_time) / (end_time - start_time) * 100
            save_alternation_stride_cycle.append(percentage)
            # print(f'percentage LESS: {percentage}')
        elif time_moments.iloc[closest_index] < alt_switch_time:
            start_time = time_moments.iloc[closest_index]
            try:
                end_time = time_moments.iloc[closest_index+1]
                percentage = (alt_switch_time - start_time) / (end_time - start_time) * 100
                save_alternation_stride_cycle.append(percentage)
                # print(f'percentage GREATER: {percentage}')
            except:
                print('IndexError: single positional indexer is out-of-bounds. The alternation happened at the end of the trial before the step cycle could be terminated.')

        # print(f'Closest index: {closest_index}')
        # print(time_moments)
        # input()

    # print(switch_data['t'].iloc[red_switch])
    # print(time_moments)
    # input()
    # plt.figure()
    # plt.plot(switch_data.index,switch_data['greenButtonPressed'])
    # plt.plot(switch_data.index,switch_data['redButtonPressed'])
    # plt.plot(switch_data.index[red_switch],switch_data['redButtonPressed'].iloc[red_switch],'*',color='r')
    # plt.plot(switch_data.index[green_switch],switch_data['greenButtonPressed'].iloc[green_switch],'*',color='g')
    # plt.show()
    return save_alternation_stride_cycle

def extract_stride_moments(track):
    sub_df = track[track['trackedObject'] == 'head'] #Extract head data
    sub_df = sub_df[sub_df['axis'] == 'y'] #Extract head bob axis
    sub_df['position'] = savgol_filter(sub_df['position'],13,2) #smooth rough tracking 
    peaks_head, _= find_peaks(sub_df['position'],distance=45)#height=np.mean(sub_df['position'])
    # vals = np.linspace(0, len(sub_df['position'])-1, len(sub_df['position']))
    # plt.plot(vals,sub_df['position'],label='head Y Position')
    # plt.scatter(vals[peaks_head],sub_df['position'].iloc[peaks_head],color='red',label='Relative Max')
    # plt.xlabel('Samples')
    # plt.ylabel('Head Y-position Height (m)')
    # plt.legend()
    # plt.show()

    # Extract data between the third and fourth peaks just for step percentage analysis - example
    # start_peak = peaks_head[2]
    # end_peak = peaks_head[3]
    # extracted_data = sub_df['position'].iloc[start_peak:end_peak + 1]
    # # Generate x-axis values
    # vals = np.linspace(0, 100, len(extracted_data))
    # # Plot the extracted data with a y-axis range from 0 to 100
    # plt.figure()
    # plt.plot(vals, extracted_data, label='head Y Position')
    # plt.xlabel('Samples')
    # plt.ylabel('Head Y-position Height (m)')
    # # plt.ylim(0, 100)  # Set y-axis range from 0 to 100
    # plt.legend()
    # plt.show()

    
    return sub_df['t'].iloc[peaks_head]

def get_each_trial(track,resp,trial):
    track_df = pd.read_csv(track)
    resp_df = pd.read_csv(resp)
    return track_df[track_df['trial'] == trial],resp_df[resp_df['trial'] == trial]

def fit_dist(df):
    per_resp = df['Percent Step Cycle'].values
    bin_edges = [i for i in range(0, 101, 10)]
    f = Fitter(per_resp, bins=bin_edges, distributions=get_common_distributions())

    f.fit()
    plt.figure()
    f.hist()
    f.plot_pdf()
    plt.ylabel('Count',fontweight='bold')
    plt.xlabel('Percent Step Cycle',fontweight='bold')
    plt.savefig('alternations_step_cycle_fitted_dist.png',dpi=400)
    plt.close()

def t_test_bins(dict_inst,bin_edges):
    """
    45 comparisons. need to apply correction for mulitple comparisons
    """
    # Bin data 
    # Extract bin edges and bin counts
    bin_edges = dict_inst[0][0]

    # Initialize bins_dict
    bins_dict = {}

    # Iterate over bin edges
    for edge in bin_edges:
        save_bins_per_edge_per_part = []
        
        # Iterate over each person's data
        for _, data in dict_inst.items():
            bin_edges, bin_counts = data
            # Use np.digitize to find the bin index
            index_val = np.digitize(edge, bin_edges, right=True) - 1
            # Check if the index is within bounds
            if 0 <= index_val < len(bin_counts):
                corresponding_data = bin_counts[index_val]
            else:
                corresponding_data = None
            save_bins_per_edge_per_part.append(corresponding_data)
        
        bins_dict[edge] = save_bins_per_edge_per_part

    # Create the DataFrame after the outer loop
    df = pd.DataFrame(bins_dict).dropna(axis=1, how='all')
    # PLOT OF ALL BARPLOTS
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    #histogram with error bars
    mean_data, std_data = df.mean(), df.sem()
    # Create an array of x-values for the bars
    # x_values = np.arange(len(mean_data))
    # ax[0].bar(x_values, mean_data, width=1, 
    #           yerr=std_data, capsize=5, tick_label=df.columns, 
    #           edgecolor='black', color='tab:blue') #x_values - bar_width / 2, mean_data
    # # ax[0].set_xlim(x_values_bar[0] - bar_width / 2, x_values_bar[-1] + bar_width / 2)
    # ax[0].set_xlim(x_values[0] - 0.5, x_values[-1] + 0.5) #remove white space
    # ax[0].set_xlabel('Percent Step Cycle', fontsize=12, fontweight='bold',labelpad=20)
    # ax[0].set_ylabel('Count', fontsize=12, fontweight='bold')

    # Define the bin edges
    bin_edges = [i for i in range(0, 101, 10)]

    # Create an array of x-values for the bars
    x_values = np.arange(len(mean_data))

    ax[0].bar(x_values, mean_data, width=1, 
            yerr=std_data, capsize=5, color='tab:blue', edgecolor='black')

    # Set the x-axis ticks and labels
    ax[0].set_xticks(x_values)
    ax[0].set_xticklabels([f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(len(bin_edges)-1)])

    ax[0].set_xlabel('Percent Step Cycle', fontsize=12, fontweight='bold', labelpad=20)
    ax[0].set_ylabel('Count', fontsize=12, fontweight='bold')

    #T-TEST FOR EACH PART OF THE STRIDE CYCLE
    columns = df.columns
    num_columns = len(columns)
    results = []
    num_tests = len(columns) * (len(columns) - 1) // 2
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            column1 = columns[i]
            column2 = columns[j]
            t_stat, p_value = ttest_rel(df[column1], df[column2])
            corrected_p_value = p_value * num_tests
            results.append((column1, column2, t_stat, corrected_p_value))

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results, columns=['Column1', 'Column2', 't_stat', 'p_value'])
    print(results_df)
    results_df = results_df.replace(np.nan, 1)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    # Pivot the results DataFrame for easy plotting
    heatmap_data = results_df.pivot(index='Column1', columns='Column2', values='p_value')
    # Create a heatmap
    sns.heatmap(ax=ax[1],data=heatmap_data, annot=True, fmt=".4f", cmap=cmap, vmin=0, vmax=0.05)
    ax[1].get_figure().set_size_inches(12, 8)
    ax[1].set_ylabel('Percent Step Cycle', fontsize=12, fontweight='bold')
    ax[1].set_xlabel('Percent Step Cycle', fontsize=12, fontweight='bold')
    x_intervals = [0, 4, 7, 9]  # Adjust as needed (0 to 100)
    y = -15  # Slightly above the x-axis label
    labels = ["Toe Off", "Heel Strike", "Midstance", "Toe Off"]

    for x, label in zip(x_intervals, labels):
        ax[0].text(x, y, label, fontsize=8, ha='center', va='top', fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_plot_grid_strep_cycle_error_bars.png',dpi=350)
    plt.close()
    # Sample p-values DataFrame (replace this with your actual data)
    # p_values_df = pd.DataFrame({
    #     0: [np.nan, 0.001039, 0.000335, 0.000980, 0.000234, 0.008933, 0.011059, 0.001978, 0.000003, 0.012197, 0.012197],
    #     10: [0.001039, np.nan, 0.106549, 0.793127, 0.004684, 0.365596, 0.260472, 0.022320, 0.002130, 0.004889, 0.004889],
    #     20: [0.000335, 0.106549, np.nan, 0.160248, 0.011819, 0.681010, 0.429568, 0.000263, 0.006572, 0.000062, 0.000062],
    #     30: [0.000980, 0.793127, 0.160248, np.nan, 0.001050, 0.192295, 0.140824, 0.113315, 0.000264, 0.025448, 0.025448],
    #     40: [0.000234, 0.004684, 0.011819, 0.001050, np.nan, 0.004302, 0.027337, 0.002222, 0.034771, 0.001206, 0.001206],
    #     50: [0.008933, 0.365596, 0.681010, 0.192295, 0.004302, np.nan, 0.112484, 0.077366, 0.085627, 0.033523, 0.033523],
    #     60: [0.011059, 0.260472, 0.429568, 0.140824, 0.027337, 0.112484, np.nan, 0.068358, 0.359018, 0.034158, 0.034158],
    #     70: [0.001978, 0.022320, 0.000263, 0.113315, 0.002222, 0.077366, 0.068358, np.nan, 0.000278, 0.000110, 0.000110],
    #     80: [0.000003, 0.002130, 0.006572, 0.000264, 0.034771, 0.085627, 0.359018, 0.000278, np.nan, 0.000105, 0.000105],
    #     90: [0.012197, 0.004889, 0.000062, 0.025448, 0.001206, 0.033523, 0.034158, 0.000110, 0.000105, np.nan, np.nan],
    #     100: [0.012197, 0.004889, 0.000062, 0.025448, 0.001206, 0.033523, 0.034158, 0.000110, 0.000105, np.nan, np.nan]
    # })
    # p_values_df.index = p_values_df.columns
    # # Number of comparisons
    # num_comparisons = 45  # Adjust the number of comparisons based on your data

    # # Calculate the Bonferroni corrected threshold
    # bonferroni_threshold = 0.05 / num_comparisons

    # # Create the heatmap with cell blocks
    # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    # t_test_results = []
    # # bins_dict = {}
    # # for _, data1 in dict_inst.items():
    # #     bin_edges, bin_counts = data1
    # #     for i, edge in enumerate(bin_edges):
    # #         if edge not in bins_dict:
    # #             bins_dict[edge] = []
    # #         bins_dict[edge].append(bin_counts[i])


    # #PLOT THE GRID OF P VALUES
    # for i in range(p_values_df.shape[0]):
    #     for j in range(p_values_df.shape[1]):
    #         p_value = p_values_df.iloc[i, j]
    #         if not np.isnan(p_value):
    #             if p_value < bonferroni_threshold:
    #                 color = 'tab:blue'
    #             else:
    #                 color = 'white'
    #             rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
    #             # rect = patches.Rectangle((j - bar_width / 2, i), bar_width, 1, linewidth=1, edgecolor='black', facecolor=color)
    #             print(f'x position: {j}')
    #             ax[1].add_patch(rect)
    # # Overlay grid lines for the second subplot
    # ax[1].grid(True, color='black', linestyle='-', linewidth=1)

    # # Set x and y axis labels for the second subplot
    # ax[1].set_xticks(range(p_values_df.shape[1]))
    # ax[1].set_xticklabels(p_values_df.columns)
    # ax[1].set_yticks(range(p_values_df.shape[0]))
    # ax[1].set_yticklabels(p_values_df.index)

    # x_intervals = [0.5, 4, 7, 9.5]  # Adjust as needed (0 to 100)
    # y = -1  # Slightly above the x-axis label
    # labels = ["Toe Off", "Heel Strike", "Midstance", "Toe Off"]

    # for x, label in zip(x_intervals, labels):
    #     ax[1].text(x, y, label, fontsize=8, ha='center', va='top', fontweight='bold')

    # # Add labels along the y-axis
    # y_intervals = [0.5, 4, 7, 9.5]  # Adjust as needed (0 to 100)
    # x_val = -0.35  # Slightly above the x-axis label
    # # labels_y = ["Label1", "Label2", "Label3"]

    # for y, label in zip(y_intervals, labels):
    #     ax[1].text(x_val, y, label, fontsize=8, ha='right', va='bottom', fontweight='bold', rotation=90)

    # ax[1].set_ylabel('Percent Step Cycle (%)',fontweight='bold',fontsize=12,labelpad=25)
    # ax[1].set_xlabel('Percent Step Cycle (%)',fontweight='bold',fontsize=12, labelpad=25) 

    # plt.tight_layout()
    # plt.savefig('multi_plot_grid_strep_cycle_error_bars.png',dpi=350)
    # plt.show()


    # Initialize a matrix for p-values
    # p_values_matrix = np.zeros((num_columns, num_columns))

    # # Perform t-tests and populate the p-values matrix
    # for i in range(num_columns):
    #     for j in range(num_columns):
    #         column1 = df[columns[i]]
    #         column2 = df[columns[j]]
    #         # print(column1)
    #         # print('=====')
    #         # print(column2)
    #         _, p_value = ttest_rel(column1, column2)
    #         # print(p_value)
    #         p_values_matrix[i, j] = p_value

    # # Create a DataFrame to display the p-values matrix
    # p_values_df = pd.DataFrame(p_values_matrix, columns=columns, index=columns)

    # print(p_values_df)
    # p_values_df = p_values_df.astype(float)
    # input()
    # # Apply Bonferroni correction
    # alpha = 0.05  # Set your desired significance level
    # n_comparisons = p_values_df.size
    # bonferroni_threshold = alpha / n_comparisons
    # significant_cells = (p_values_df < bonferroni_threshold) & (p_values_df > 0)

    # # Create a heatmap with significant cells highlighted in light red using Matplotlib
    # plt.figure(figsize=(12, 10))
    # heatmap = plt.imshow(p_values_df, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # # Display p-values within significant cells
    # for i in range(p_values_df.shape[0]):
    #     for j in range(p_values_df.shape[1]):
    #         p_value = p_values_df.iloc[i, j]
    #         if significant_cells.iloc[i, j]:
    #             plt.text(j, i, f'{p_value:.6f}', ha='center', va='center', color='black')

    # # Highlight significant cells in light red
    # significant_cells = significant_cells.astype(float)
    # heatmap = plt.imshow(significant_cells, cmap='gray', aspect='auto', vmin=0, vmax=1, extent=[-0.5, significant_cells.shape[1] - 0.5, -0.5, significant_cells.shape[0] - 0.5])

    # plt.title('P-Values Heatmap (Bonferroni Corrected)')
    # # plt.colorbar(heatmap, orientation='vertical', label='P-Values')
    # # plt.grid(True, linestyle='--', linewidth=0.5, color='black', which='both')

    # # Label the axes with column names (0-100)
    # plt.xticks(range(p_values_df.shape[1]), p_values_df.columns)
    # plt.yticks(range(p_values_df.shape[0]), p_values_df.index)

    # plt.show()
    # column_headers = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # row_headers = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # # Number of comparisons
    # # Number of comparisons
    # num_comparisons = 45

    # # Calculate the Bonferroni corrected threshold
    # bonferroni_threshold = 0.05 / num_comparisons

    # # Apply Bonferroni correction to your p-values
    # corrected_df = p_values_df * num_comparisons

    # # Create a Seaborn heatmap
    # plt.figure(figsize=(10, 8))

    # # Create a mask for p-values less than the threshold
    # mask = corrected_df < bonferroni_threshold

    # # Create a custom color palette
    # colors = sns.light_palette("blue", as_cmap=True)

    # # Create the heatmap with p-value annotations
    # sns.heatmap(corrected_df, annot=True, fmt=".5f", cmap=colors, cbar=False, xticklabels=True, yticklabels=True, mask=~mask)

    # plt.show()

def main():
    #standing alternation and mixed percepts
    csv_stand_files = read_stand_files()
    for i, resp in enumerate(csv_stand_files):
        for trial in range(1,9):
            print(f'trial: {trial}')
            resp_df = pd.read_csv(resp)
            trial_df = resp_df[resp_df['trial'] == trial]
            alt_rate_stand(trial_df)
    #Walk analysis
    csv_files_walk, csv_files_track = read_all_files()
    save_all_alt_stride = []
    # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # fig.suptitle("Histograms for Each Participant", fontsize=16)  
    bin_count_subject = {}
    combined_df = pd.DataFrame(columns=['Percent Step Cycle'])  
    for i, (resp, track) in enumerate(zip(csv_files_walk, csv_files_track)):
        for trial in range(1,9):
            print(f'trial: {trial}')
            track_df, resp_df = get_each_trial(track,resp,trial)
            time_moments = extract_stride_moments(track_df)
            save_all_alt_stride.append(alt_rate_try(resp_df,time_moments))
            # save_all_alt_stride.append(calc_alternation_location(resp_df,time_moments))
        df = pd.DataFrame([item for sublist in save_all_alt_stride for item in sublist],
                    columns=['Percent Step Cycle'])
        df.sort_values(by=['Percent Step Cycle'],inplace=True)
        df = df[df['Percent Step Cycle'] != np.inf]

        # # Plot with 10% bins in the appropriate subplot
        bin_edges = [i for i in range(0, 101, 10)]
        # row, col = divmod(i, 3)  # Calculate the row and column for the subplot
        # ax = axes[row, col]
        # sns.histplot(data=df, x="Percent Step Cycle", bins=bin_edges, kde=True, ax=ax)
        # ax.set_ylabel('Count', fontweight='bold')
        # ax.set_xlabel('Percent Step Cycle', fontweight='bold')
        bin_counts = np.histogram(df["Percent Step Cycle"], bins=bin_edges)[0]
        bin_count_subject[i] = [bin_edges, bin_counts]
        # # print(bin_count_subject)
        # ax.set_title(f"Participant {i + 1}",fontweight='bold')
        combined_df = pd.concat([combined_df, df])
    plt.tight_layout()
    plt.savefig('alternations_step_cycle_first_all_participants.png',dpi=400)
    plt.close()
    # plt.show()
    
    #COMBINED PLOT
    combined_df.sort_values(by=['Percent Step Cycle'], inplace=True)
    bin_edges = [i for i in range(0, 101, 10)]
    #t-test compare bin counts
    t_test_bins(bin_count_subject,bin_edges)
    # sns.histplot(data=combined_df, x="Percent Step Cycle", bins=bin_edges, kde=True)
    # plt.ylabel('Count',fontweight='bold')
    # plt.xlabel('Percent Step Cycle',fontweight='bold')
    # plt.tight_layout()
    # plt.savefig('alternations_step_cycle_first_combined.png',dpi=400)
    # plt.close()

    #check normality of the data
    # stat, p = shapiro(df['Percent Step Cycle'])

    # Perform the Kolmogorov-Smirnov test
    _, ks_p_value = kstest(df['Percent Step Cycle'], norm.cdf)
    if ks_p_value < 0.05:
        print("The data significantly differs from a normal distribution.")
    else:
        print("The data does not significantly differ from a normal distribution.")

    _, ks_p_value_uniform = kstest(df['Percent Step Cycle'], uniform.cdf)
    if ks_p_value_uniform < 0.05:
        print("The data significantly differs from a uniform distribution.")
    else:
        print("The data does not significantly differ from a uniform distribution.")

    #qqplot normal distribution
    # plt.figure()
    # sm.qqplot(combined_df['Percent Step Cycle'], line='s')
    # plt.title("Q-Q Plot")
    # plt.savefig('qq_plot_all_resp_stride.png',dpi=400)
    # plt.close()

    #find best hist
    # fit_dist(combined_df)

if __name__ == "__main__":
    main()