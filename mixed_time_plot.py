import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
"""
the mean percentage is 2.83%
"""
def hist_mixed():
    df = pd.read_csv('mixed_percentage_per_trial.txt', sep='\t', header=None)
    print(f'{df.mean()} +/-  {df.std()}')
    df.columns = ['mixed_time']
    sns.histplot(df['mixed_time'], bins=40, kde=False, color='tab:blue')
    # custom_xticks = range(0, int(max(df['mixed_time'])), 20)  # Adjust range and step size as needed
    # plt.xticks(custom_xticks)
    # plt.title('Distribution of mixed_time')   
    plt.xlabel('Percentage of Trial Duration with Mixed Perception',fontweight='bold',fontsize=12)
    plt.ylabel('Count',fontweight='bold',fontsize=12)
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.tight_layout()
    plt.savefig('mixed_histogram.png',dpi=350)
def main():
    hist_mixed()
if __name__ == "__main__":
    main()