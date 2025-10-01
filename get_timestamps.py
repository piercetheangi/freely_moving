#timestamps for stims
import pandas as pd
import glob
from pathlib import Path

def add_video_data(df, video_name, start_times, end_times, states):
    data = {
        'Video': [video_name]*len(start_times), 
        'Start Time (s)': start_times, 
        'End Time (s)': end_times, 
        'State': states
    }
    df_new = pd.DataFrame(data)
    return pd.concat([df, df_new], ignore_index=True)

# Filter all .avi video files for one subject
subject = '17' # ---------------------------------------------------------------------- SPECIFY SUBJECT TO ANALYSE HERE 
inference_folder = f'/Volumes/Data/{subject}'

# Extract file names
video_files = glob.glob(f"{inference_folder}/{str(subject)}*.h5")
file_names = [Path(f).stem for f in video_files]
print("Found video files:", file_names)

# Create an empty DataFrame to hold all the data
dftime = pd.DataFrame(columns=['Video', 'Start Time (s)', 'End Time (s)', 'State'])

# Stim parameters
# Default setting: First stim at 15sec. All stim duration 20sec. 60sec interval between each stim train. 
start_times = [0, 15, 35, 95, 115, 175, 195] # Default
end_times = [15, 35, 95, 115, 175, 195, 255]
states = ['OFF', 'ON', 'OFF', "ON", "OFF", "ON", "OFF"]

for i in file_names:
    dftime = add_video_data(dftime, i, start_times, end_times, states)
    
print(dftime)

# Write the main DataFrame to a .csv file
dftime.to_csv('/Volumes/Data/NM_timestamps.csv', mode='a', index=False, header=False) # Appends processed timestamps to existing .csv file
# dftime.to_csv('/Volumes/Data/timestamps.csv', mode='a', index=False, header=False) # AY subjects

# Double-check
# test = pd.read_csv('/Volumes/Data/NM_timestamps.csv') # Loads .csv file back in
# np.unique(test['Video'])
