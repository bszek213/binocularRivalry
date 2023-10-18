import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks,find_peaks_cwt, savgol_filter
import os
import glob
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

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
    return sub_df['t'].iloc[peaks_head]

def get_each_trial(track,resp,trial):
    track_df = pd.read_csv(track)
    resp_df = pd.read_csv(resp)
    return track_df[track_df['trial'] == trial],resp_df[resp_df['trial'] == trial]

def main():
    csv_files_walk, csv_files_track = read_all_files()
    save_all_alt_stride = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Histograms for Each Participant", fontsize=16)  
    for i, (resp, track) in enumerate(zip(csv_files_walk, csv_files_track)):
        for trial in range(1,9):
            print(f'trial: {trial}')
            track_df, resp_df = get_each_trial(track,resp,trial)
            time_moments = extract_stride_moments(track_df)
            save_all_alt_stride.append(alt_rate_try(resp_df,time_moments))
            # save_all_alt_stride.append(calc_alternation_location(resp_df,time_moments))
        df = pd.DataFrame([item for sublist in save_all_alt_stride for item in sublist],
                    columns=['Percent Step Cycle'])
        # Plot with 10% bins in the appropriate subplot
        bin_edges = [i for i in range(0, 101, 10)]
        row, col = divmod(i, 3)  # Calculate the row and column for the subplot
        ax = axes[row, col]
        sns.histplot(data=df, x="Percent Step Cycle", bins=bin_edges, kde=True, ax=ax)
        ax.set_title(f"Participant {i + 1}")
    plt.tight_layout()
    plt.savefig('alternations_step_cycle_first_all_participants.png',dpi=400)
    plt.show()
if __name__ == "__main__":
    main()