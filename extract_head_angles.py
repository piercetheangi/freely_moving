import sys
sys.path.append('/Users/feiyang/Projects/GLM-HMM')
sys.path.append('/Users/feiyang/Projects/GLM-HMM/ssm')
sys.path.append('/Users/feiyang/Projects/Reverse')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import glob
import h5py
import re
from psychometric_utils import moving_average, interpolate_array, calculate_head_angles, calculate_total_rotations, average_sessions, process_head_angles

""" KEY DATAFRAMES
timestamps: timestamps data retrieved from timestamps for stim.py
head_rotations: contains total head rotations data for all subjects
all_head_angles: contains session-by-session continuous head angle data for all subjects
"""

#### IMPORTANT ####
# 83_R_snr_ctr: missing a large chunk of frames (mouse covered), make sure to not interpolate this  # actually i think this is fine.
###### ^^^^^ ######

# For renaming files
import os
from pathlib import Path
folder = Path("/Volumes/Data/17")

for file in folder.iterdir():
    if file.is_file():
        stem = file.stem                    # Remove suffix. e.g. '88_R_50hz.h5' -> 88_R_50hz
        parts = stem.split('_')             # Split by '_'

        if len(parts) == 3:                 # If file name is missing brain region...
            parts.insert(2, 'VLS')          # Insert 'VLS'/**BRAIN_REGION** after the second part
       
        new_stem = '_'.join(parts)          # Join back with '_'
        new_name = new_stem + file.suffix   # Add suffix back

        # Replace left/right
        new_stem = new_stem.replace('left', 'L').replace('right', 'R')

        if new_stem != stem:
            new_name = new_stem + file.suffix
            new_file = file.with_name(new_name)            # Build full new path

            file.rename(new_file)                           # Rename
            print(f"Renamed: {file.name} -> {new_name}")


# Load the timestamps DataFrame
dftime = pd.read_csv('/Volumes/Data/NM_timestamps.csv') # Timestamps data
# head_rotations = pd.read_csv('/Users/feiyang/Projects/GLM-HMM/head_rotations.csv') # Head rotations data              # AY SUBJECTS
head_rotations = pd.read_csv('/Volumes/Data/SLEAP_Project_final/inference_mouse_data.csv')                              # NM SUBJECTS

FRAMERATE = 25

all_head_angles = {}

num = 99
subject = f'SWC_NM_0{num}'                                                                                                   # SPECIFY SUBJECT FOR ANALYSIS

if num in np.unique(head_rotations['Mouse number']):
    print(f'Head rotational data for {subject} already exists. Continuing would overwrite the existing data.')
    input("Press Enter to continue...")

else: 
    # head_rotations[subject] = {}
    all_head_angles[subject] = {}


confidence_threshold = 0.8  # Frames below this confidence score are removed and values are interpolated.                # SETTING #
failed_videos = []

# Find all video files belonging to that subject
folder_path = f'/Volumes/Data/{num}'
h5_files = glob.glob(f"{folder_path}/{str(num)}*.h5")  
file_names = [file.split('/')[-1] for file in h5_files]

print(f'Found files: {file_names}')

# Loops through all video files
for j, vid in enumerate(h5_files):
    FILE_PATH = vid

    print(f'Analysing video: {file_names[j]}...')

    ##### Load data
    # with h5py.File(FILE_PATH, "r") as f:
    #     print(list(f.keys()))  # Prints the top-level structure of the file
    with h5py.File(FILE_PATH, 'r') as f:

        occupancy_matrix = f['track_occupancy'][:].T
        tracks_matrix = f['tracks'][:].T  # Matrix containing (Nframes, Nnodes, xyCoordinates)
        node_names = [name.decode('utf-8') for name in f['node_names'][:]]
        instance_scores = np.array(f["instance_scores"]).T  # Confidence score for individual nodes

        # print(node_names)
        f.close()

    ##### Remove frames with low confidence score
    flagged_frames = np.where(instance_scores < confidence_threshold)[0]
    filt_tracks_matrix = np.copy(tracks_matrix)
    filt_tracks_matrix[flagged_frames, :, :, :] = np.nan

    print(f'{len(flagged_frames)} / {tracks_matrix.shape[0]} frames flagged for deletion.')

    ##### Load corresponding video timestamps
    current_video = FILE_PATH.split('/')[-1].replace('.h5', '')
    filtered_timestamps = dftime[dftime['Video'] == current_video]

    if filtered_timestamps.empty:
        print(f"Insufficient timestamps found for video {file_names[j]}. Skipping...")
        failed_videos.append(file_names[j])
        continue

    ##### Loop through each OFF/ON state 
    rotations = []
    avg_rotation_time = []
    rotation_per_20s = []
    session_angles = []
    control_counter = 0
    session_counter = 0
    for i in range(len(filtered_timestamps)): 

        curr_period = filtered_timestamps.iloc[i]
        state = curr_period['State']                                
        start_time = curr_period['Start Time (s)']
        end_time = curr_period['End Time (s)']
        segment_time = end_time - start_time

        start_index = int(start_time * FRAMERATE) # frame index
        end_index = int(end_time * FRAMERATE)

        timestamps = np.arange(start_index, end_index) / FRAMERATE  # Convert frame indices to time (seconds)
        framestamps = np.array(range(0, len(timestamps))) + start_index # use as x-axis in place of timestamp. 

        #print(f'Analysing time frame: {start_time} - {end_time}sec, stim state: {state}')

        # Head angle calculated using nose-neck vector
        neck_x, neck_y = filt_tracks_matrix[start_index:end_index, 1, 0, 0], filt_tracks_matrix[start_index:end_index, 1, 1, 0] # Node 1
        nose_x, nose_y = filt_tracks_matrix[start_index:end_index, 5, 0, 0], filt_tracks_matrix[start_index:end_index, 5, 1, 0] # Node 5

        nodes = {
            "neck": (neck_x, neck_y),
            "nose": (nose_x, nose_y),
        }

        # Interpolate arrays
        interp_nodes = {}
        for node_name, (x, y) in nodes.items():
            interp_nodes[node_name] = interpolate_array(x, y)

        # Extract interpolated arrays
        neck_x_interp, neck_y_interp = interp_nodes["neck"]
        nose_x_interp, nose_y_interp = interp_nodes["nose"]

        # Calculate head angle in radians
        head_angles = calculate_head_angles(interp_nodes, angle_type='radian')
        corrected_angles = np.unwrap(head_angles)  # Removes angle discontinuity artefacts (.unwrap can only be applied to radians!!)
        session_angles.append(head_angles)

        # Calculate total rotations
        total_rotations, cw_rotation, ccw_rotation = calculate_total_rotations(corrected_angles, smoothing_window=10) 
        avg_rotation_time = segment_time / total_rotations if total_rotations != 0 else 0     
        rotation_per_20s = total_rotations / segment_time * 20 if segment_time != 0 else 0

        cw_rotation_per_20s = cw_rotation / segment_time * 20 if segment_time != 0 else 0
        ccw_rotation_per_20s = ccw_rotation / segment_time * 20 if segment_time != 0 else 0

        ##### Storing data
        name_components = current_video.split('_')
        file_name = FILE_PATH.split('/')[-1].replace('.h5', '')

        # try: 
        #     video_num = name_components[4]
        # except IndexError as e: 
        #     video_num = '1'

        if curr_period["State"] == "OFF":
            control_counter += 1  # Increment OFF counter
            session_label = ""  # Empty session entry
        else:  # ON trial
            session_counter += 1  # Increment ON counter
            control_label = ""  # Empty control entry

        # Dictionary for storing data 
        video_info = {
            'Frames': len(head_angles),  # Total frames
            'Total Rotations': total_rotations,  # Total rotations
            'Average time per rotation (s)': avg_rotation_time,  # Average time per rotation
            'Rotations per 20 seconds': rotation_per_20s,  # Rotations per 20 seconds
            'Mouse number': int(name_components[0]),  # Mouse number
            'Hemisphere': name_components[1].capitalize(),  # Hemisphere
            'Brain region': name_components[2],  # Brain region
            'Stimulation': name_components[3],  # Stimulation
            f'Time stim is {curr_period["State"]} (s)': segment_time,  # Time stimulus is ON or OFF
            'Session #': session_counter if curr_period["State"] == "ON" else "",  # Assign session number for ON
            'Control #': control_counter if curr_period["State"] == "OFF" else "",
            'File name path': FILE_PATH, 
            'Video file': file_name
        }

        head_rotations = pd.concat([head_rotations, pd.DataFrame([video_info])], ignore_index=True)

    all_head_angles[subject][file_name] = session_angles

##### Save data to .csv file
# head_rotations.to_csv('/Volumes/Data/SLEAP_Project_final/inference_mouse_data.csv', index=False)              # NM Subjects

##### Save data to .csv file
# head_rotations.to_csv('/Users/feiyang/Projects/GLM-HMM/CONFIRMCORRECThead_rotations.csv', index=False)        # AY Subjects


# with open('/Users/feiyang/Projects/GLM-HMM/CONFIRMCORRECTall_head_angles.pkl', 'wb') as file:
#     pickle.dump(all_head_angles, file)

# Double check
# test = pd.read_csv('/Users/feiyang/Projects/GLM-HMM/head_rotations.csv') 
# with open('/Users/feiyang/Projects/GLM-HMM/all_head_angles.pkl', "rb") as file: 
#     test_angles = pickle.load(file)

# SNr: 2, 6, 8, 12

# ##### Plot continuous head angles 
with open('/Users/feiyang/Projects/GLM-HMM/all_head_angles.pkl', "rb") as file: 
    all_head_angles = pickle.load(file)

# Average across subjects
_, averaged_angles, sessions_std = average_sessions(all_head_angles, merge_repeats=True, normalisation=True) 

# Extract start & end indices for each block
example_session = all_head_angles['SWC_AY_012']['12_right_snr_0hz_2']
start_indices = np.cumsum([0] + [len(block) for block in example_session[:-1]])       
end_indices = start_indices + np.array([len(block) for block in example_session]) - 1

# # Can hardcode if stim ON/OFF durations have not been changed. 
# start_indices = [0, 375, 875, 2375, 2875, 4375, 4875]
# end_indices = [374, 874, 2374, 2874, 4374, 4874, 6374]

# Plot AVERAGED head angles across session
session_type = 'right_snr_0hz'
angles_to_plot = averaged_angles[session_type]
angles_to_plot = process_head_angles(angles_to_plot, 
                                     cutoff_freq=4.0, 
                                     sampling_rate=100, 
                                     baseline_correction=True) # Applies low-pass filter & baseline correction 
std_to_plot = sessions_std[session_type]

plt.figure(figsize=(10, 5))

upper_bound = angles_to_plot + std_to_plot
lower_bound = angles_to_plot - std_to_plot

plt.plot(range(len(angles_to_plot)), angles_to_plot, 
         color="navy", linewidth=2, alpha=0.6)

plt.fill_between(range(len(angles_to_plot)), lower_bound, upper_bound, # Standard deviation
                 color="navy", alpha=0.2)

# Shade stim ON chunks
for i in range(len(start_indices)):
    if i % 2 == 1: 
        plt.axvspan(start_indices[i], end_indices[i], alpha=0.2, edgecolor="none")

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Time (sec.)")
plt.ylabel("Total Clockwise Rotation (Norm. rad.)")

plt.title("Head Orientation")
plt.show()

# Sample size


"""
    plt.text(0.02, 0.95, f"Total mice: {num_mice}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.90, f"Total sessions: {num_sessions}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.85, f"Total controls: {num_controls}", transform=plt.gca().transAxes)
    print(f'{brain_region}: \nN mice: {num_mice} \nN sessions: {num_sessions} \nN controls: {num_controls}')
"""

# Plot head angles across INDIVIDUAL session
                                                                # NEED TO IMPLEMENT Z-SCORE NORMALISATION TO INDIVIDUAL DATA 
subject = 'SWC_AY_012'
session_type = '12_right_snr_0hz_2'
session_angles = all_head_angles[subject][session_type]

continuous_angles = np.concatenate(session_angles)  # Collapses to a continuous session
angles_to_plot = np.unwrap(continuous_angles)  # Removes angle discontinuity

plt.figure(figsize=(10, 5))
plt.plot(range(len(angles_to_plot)), angles_to_plot, label="Head Angle", 
        color="navy", linewidth=1.5, alpha=0.6)

#plt.plot(range(len(angles_to_plot)), angles_to_plot, label="Head Angle", 
#         color="navy", linewidth=1.5, alpha=0.6)

# Shade stim ON chunks
for i in range(len(start_indices)):
    if i % 2 == 1: 
        plt.axvspan(start_indices[i], end_indices[i], alpha=0.2, edgecolor="none")

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Time (sec.)")
plt.ylabel("Total Clockwise Rotation (rad.)")
plt.title("Head Orientation")
plt.legend()
plt.show()



# # Visualise quiver plot
# dx = np.cos(corrected_angles)
# dy = np.sin(corrected_angles)

# plt.figure(figsize=(7, 5))  
# plt.quiver(nose_x_interp, nose_y_interp, dx, dy, corrected_angles,
#            angles='xy', scale_units='xy', 
#            scale=0.1, width=0.02, cmap='turbo')

# plt.colorbar(label="Time (sec.)")  # Add colorbar
# plt.axis('equal')  # to keep the arrows consistent in size
# plt.gca().invert_yaxis()  # invert the y-axis
# plt.show()





