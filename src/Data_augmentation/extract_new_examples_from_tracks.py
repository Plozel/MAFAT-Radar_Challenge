import os
import pickle
import numpy as np
import pandas as pd


# Choose data:
data_folder = '/home/yonattan/projects/MAFAT-Radar_Challenge/data'
set_name = 'MAFAT RADAR Challenge - Training Set V1'

# Read data:
metadata_df = pd.read_csv(os.path.join(data_folder, set_name + '.csv'))
with open(os.path.join(data_folder, set_name + '.pkl'), 'rb') as f:
    data = pickle.load(f)

# Init new data structures:
new_metadata = []
new_data = {'doppler_burst': [],
            'iq_sweep_burst': []}

# Iterate over the dataset and append to the new dataset:
metadata_df.set_index('segment_id', inplace=True, verify_integrity=True)
for track_id in metadata_df['track_id'].unique():
    segment_id_list = metadata_df.loc[metadata_df['track_id'] == track_id].index.to_list()
    for segment_id in segment_id_list:
        next_segment_id = segment_id + 1
        if next_segment_id not in segment_id_list:
            continue
        if not metadata_df.loc[segment_id].equals(metadata_df.loc[next_segment_id]):
            continue
        new_metadata.append(metadata_df.loc[segment_id].to_dict())
        # Splice the data:
        doppler_burst = data['doppler_burst'][segment_id, :]
        doppler_burst_next = data['doppler_burst'][next_segment_id, :]
        doppler_burst_new = np.concatenate((doppler_burst[16:], doppler_burst_next[:-16]))
        new_data['doppler_burst'].append(doppler_burst_new)
        iq_sweep_burst = data['iq_sweep_burst'][segment_id, :, :]
        iq_sweep_burst_next = data['iq_sweep_burst'][next_segment_id, :, :]
        iq_sweep_burst_new = np.concatenate((iq_sweep_burst[:, 16:], iq_sweep_burst_next[:, :-16]), axis=1)
        new_data['iq_sweep_burst'].append(iq_sweep_burst_new)

# Rearrange the new data and add segment_id:
new_data['iq_sweep_burst'] =  np.stack(new_data['iq_sweep_burst'])
new_data['doppler_burst'] = np.stack(new_data['doppler_burst'])
new_data['segment_id'] = np.arange(new_data['doppler_burst'].shape[0])
new_metadata = pd.DataFrame(new_metadata)
new_metadata['segment_id'] = new_metadata.index

# Save the new set:
new_set_name = set_name + ' - Spliced'
new_metadata.to_csv(os.path.join(data_folder, new_set_name + '.csv'))
with open(os.path.join(data_folder, new_set_name + '.pkl'), 'wb') as f:
    pickle.dump(new_data, f)

print('Created {} new examples! Saved the data to: {}'.format(new_data['doppler_burst'].shape[0], os.path.join(data_folder, new_set_name)))