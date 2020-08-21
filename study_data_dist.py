import os
import numpy as np
import pandas as pd


datasets={
        "train": "MAFAT RADAR Challenge - Training Set V1",
        "train_spliced": "MAFAT RADAR Challenge - Training Set V1 - Spliced",
        "experiment": "MAFAT RADAR Challenge - Auxiliary Experiment Set V2",
        "synthetic": "MAFAT RADAR Challenge - Auxiliary Synthetic Set V2",
        "test": "MAFAT RADAR Challenge - Public Test Set V1",
        "empty": "MAFAT RADAR Challenge - Auxiliary Background(empty) Set V1",
        }

data_folder_path = 'Data'

df_list = []
for set_name, file_name in datasets.items():
    if set_name == 'test':
        continue
    df = pd.read_csv(os.path.join(data_folder_path, file_name + '.csv'))
    df['set_name'] = set_name
    df['val'] = int(1)
    df_list.append(df)
concatenated_df = pd.concat(df_list)

# Create pivot tables:
pt_target_type_vs_geolocation_type_and_id = pd.pivot_table(concatenated_df, values='val', index='target_type', columns=['geolocation_type', 'geolocation_id'], aggfunc=np.sum)
pt_target_type_vs_geolocation_type = pd.pivot_table(concatenated_df, values='val', index='target_type', columns='geolocation_type', aggfunc=np.sum)
pt_target_type_vs_sensor_id = pd.pivot_table(concatenated_df, values='val', index='target_type', columns='sensor_id', aggfunc=np.sum)
pt_geolocation_vs_sensor_id = pd.pivot_table(concatenated_df, values='val', index='geolocation_type', columns='sensor_id', aggfunc=np.sum)
pt_geolocation_and_target_type_vs_sensor_id = pd.pivot_table(concatenated_df, values='val', index=['geolocation_type', 'target_type'], columns='sensor_id', aggfunc=np.sum)

x = 0
