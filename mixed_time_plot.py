import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
"""
2.8312786127856127 +/-  2.5178274753713783
2.8140595065091603 +/-  3.043417582798697

"""
def hist_mixed():
    #walk
    df = pd.read_csv('mixed_percentage_per_trial_walk.txt', sep='\t', header=None)
    
    #stand
    df_stand = pd.read_csv('mixed_percentage_per_trial_stand.txt', sep='\t', header=None)
    
    df.columns = ['mixed_time']
    df_stand.columns = ['mixed_time']
    print(f"{df['mixed_time'].mean()} +/-  {df['mixed_time'].std()}")
    print(f"{df_stand['mixed_time'].mean()} +/-  {df_stand['mixed_time'].std()}")
    # min_length = min(len(df_stand), len(df_stand))
    # df1_truncated = df_stand.head(min_length)
    # df2_truncated = df_stand.head(min_length)
    # combined_df = pd.concat([df1_truncated, df2_truncated], ignore_index=True)

    # df['category'] = ['Walk'] * len(df)
    # df_stand['category'] = ['Stand'] * len(df_stand)

    combined_df = pd.DataFrame({"Stand":df_stand['mixed_time'].iloc[0:len(df)], "Walk":df['mixed_time']})

    sns.set(style="whitegrid")
    plt.figure(figsize=(4, 6))
    ax = sns.pointplot(data=combined_df,
            kind="point", color='tab:blue',capsize=.1,ci=95,join=True, markers='o',scale=0.8, dodge=False)
    ax.scatter(x=['Stand']*len(df_stand), y=df_stand['mixed_time'], 
               color='tab:orange', s=40, marker='o',alpha=0.6)
    ax.scatter(x=['Walk']*len(df), y=df['mixed_time'], 
               color='tab:orange', s=40, marker='o',alpha=0.6)
    plt.ylabel('Mixed Percepts per Trial (%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('Mixed_percepts_per_trial',dpi=350)
    # sns.histplot(df['mixed_time'], bins=40, kde=False, color='tab:blue')
    # custom_xticks = range(0, int(max(df['mixed_time'])), 20)  # Adjust range and step size as needed
    # plt.xticks(custom_xticks)
    # plt.title('Distribution of mixed_time')   
    # plt.xlabel('Percentage of Trial Duration with Mixed Perception',fontweight='bold',fontsize=12)
    # plt.ylabel('Count',fontweight='bold',fontsize=12)
    # plt.xticks(weight='bold')
    # plt.yticks(weight='bold')
    # 
    # plt.savefig('mixed_histogram.png',dpi=350)
def main():
    hist_mixed()
if __name__ == "__main__":
    main()