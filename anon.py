#anon
import pandas as pd
import os

anon_df = pd.read_csv('binocular_rivalry_master.csv')
print(anon_df)
for directory in os.listdir(os.path.join(os.getcwd(),"Data")): #participant level
    for dir2 in os.listdir(os.path.join(os.getcwd(),"Data",directory)): #conditions
        for files in os.listdir(os.path.join(os.getcwd(),"Data",directory,dir2)):
            
            # for old_name in anon_df['name']:
            #     if old_name in files:
            #         # get the corresponding new name
            #         new_name = anon_df.loc[anon_df['name']==old_name, 'new_name'].values[0]
                    
            #         # construct the new filename by replacing the old name with the new name
            #         new_filename = files.replace(old_name, new_name)
            #         # rename the file
            #         print(os.path.join(os.getcwd(),"Data",directory,dir2,files))
            #         directory_full = os.path.join(os.getcwd(),"Data",directory,dir2)
            #         os.rename(os.path.join(directory_full, files), 
            #                   os.path.join(directory_full, new_filename))
            #         input()
    # # check if the directory name is present in the dataframe
    # if directory in anon_df['name'].values:
    #     # get the corresponding new name
    #     new_name = anon_df.loc[anon_df['name']==directory, 'new_name'].values[0]
    #     print(directory)
    #     print(new_name)
    #     input()
    #     # rename the directory
    #     os.rename(os.path.join(os.getcwd(), 'Data', directory), 
    #               os.path.join(os.getcwd(), 'Data', str(new_name)))