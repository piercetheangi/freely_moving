import sys
import pickle
sys.path.append('/Users/feiyang/Projects/GLM-HMM')
sys.path.append('/Users/feiyang/Projects/GLM-HMM/ssm')
sys.path.append('/Users/feiyang/Projects/Reverse')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind
from psychometric_utils import annotate_p_value

# pd.set_option('display.max_columns', 50)                                  # Forces pd. to display up to 50 columns

# Read the DataFrame from the CSV file
df = pd.read_csv('/Volumes/Data/SLEAP_Project_final/inference_mouse_data.csv')                  # SWC_NM_ subjects
# df = pd.read_csv('/Users/feiyang/Projects/GLM-HMM/head_rotations.csv')                        # SWC_AY Head rotations data
df_pvals = pd.read_csv("/Volumes/Data/p_values.csv")                                            # p-vals for all L/R vs. CTR t-tests (REFER TO rotations master plot.py script)

df['Stimulation'] = df['Stimulation'].str.upper()
df.columns = df.columns.str.replace(' ', '_')

# Define brain regions, stimulations, and hemispheres
brain_regions = ['VLSD2'] # df['Brain_region'].unique().tolist()
stimulations = ['0HZ', '50HZ'] # df['Stimulation'].unique().tolist() 
hemispheres = ["Left", "Right", "Control"]


by_individual_mouse = 1                                                                         # OPTION TO VIEW MOUSE-BY-MOUSE
mouseID = 83

# VLS D2 subjects: [83, 87, 89, 88, 42]
# VLS D1 subjects: [93, 99, 38]


colors_dict = {
    'Left': {
        '0HZ': 'lightblue',
        '50HZ': 'lightblue',
        '2HZ': '#A1CAF1',
        '4HZ': '#0000FF',
        'stat_significant': '#337CCF'
    },
    'Right': {
        '0HZ': 'lightpink',
        '50HZ': 'lightpink',
        '2HZ': '#FBB982',
        '4HZ': '#FF6103',
        'stat_significant': '#BB2525'
    }
}


p_values_dict = {}
# Loop through each brain region
for brain_region in brain_regions:

    sub_df_all = df[df['Brain_region'] == brain_region].copy()

    if by_individual_mouse == 1: # Analyse by individual mouse
        this_mouse = np.logical_and(df['Brain_region'] == brain_region,
                                    df['Mouse_number'] == mouseID)
        sub_df_all = df[this_mouse].copy()

    sub_df_all['Avg_Rotations_per_20_seconds'] = sub_df_all['Total_Rotations'] / (sub_df_all['Frames'] / 500)

    num_mice = sub_df_all['Mouse_number'].nunique()
    num_sessions = sub_df_all['Session_#'].count()
    num_controls = sub_df_all['Control_#'].count()

    if sub_df_all.empty:
        print(f"No data for {brain_region}, skipping...")
        continue

    mean_values = []
    sem_values = []
    x_positions = []
    colors = []
    labels = []

    control_means_by_stim = {}

    
    bar_position = 0
    gap_between_groups = 0.2
    for stim in stimulations:
        if stim not in sub_df_all['Stimulation'].values:
            print(f"No data for {stim} in {brain_region}, skipping...")
            continue

        control_means = []
        control_sems = []

        for hemi in hemispheres:
            print(f"Processing {brain_region}, {hemi}, {stim}")  # Debug statement
            for condition_column in ['Control_#', 'Session_#']:
                sub_df = sub_df_all[(sub_df_all['Stimulation'] == stim) &
                                    (sub_df_all['Hemisphere'] == hemi) &
                                    pd.notna(sub_df_all[condition_column])].copy()
                print(f"Length of sub_df for {brain_region}, {hemi}, {stim}: {len(sub_df)}")

                if sub_df.empty:
                    continue

                sub_df['Avg_Rotations_per_20_seconds'] = sub_df['Total_Rotations'] / (sub_df['Frames'] / 500)
                mean_rotations = np.mean(sub_df['Avg_Rotations_per_20_seconds'])
                sem_rotations = stats.sem(sub_df['Avg_Rotations_per_20_seconds'])

                if 'Control' in condition_column:
                    control_means.append(mean_rotations)
                    control_sems.append(sem_rotations)
                else:
                    mean_values.append(mean_rotations)
                    sem_values.append(sem_rotations)
                    x_positions.append(bar_position)
                    labels.append(f"{hemi} ({stim})")
                    colors.append(colors_dict[hemi][stim])  # Append the color right after appending the label
                    bar_position += 0.1

                # Insert the print statements here
                if brain_region == "VLSD2" and hemi == "Right" and stim == "50HZ":
                    print("Mean Values:", mean_values)
                    print("SEM Values:", sem_values)
                    print("X Positions:", x_positions)
                    
        avg_control = np.mean(control_means)
        avg_control_sem = np.mean(control_sems)
        control_means_by_stim[stim] = avg_control
        mean_values.append(avg_control)
        sem_values.append(avg_control_sem)
        x_positions.append(bar_position)
        labels.append(f"Control")
        colors.append('#D3D3D3')  # Color for Control
        bar_position += gap_between_groups
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_positions, mean_values, color=colors, width=0.1, yerr=sem_values, capsize=5, error_kw={'elinewidth': 1})
    max_val = max(mean_values)

    # Local dictionary to store p-values for the current brain region
    # ----- 
    local_p_values = {}

    if '0HZ' in sub_df_all['Stimulation'].values:
        # Comparing Left vs. Right Hemisphere for 0Hz
        left_0hz_data = sub_df_all[(sub_df_all['Stimulation'] == '0HZ') & (sub_df_all['Hemisphere'] == 'Left')]['Avg_Rotations_per_20_seconds']
        right_0hz_data = sub_df_all[(sub_df_all['Stimulation'] == '0HZ') & (sub_df_all['Hemisphere'] == 'Right')]['Avg_Rotations_per_20_seconds']

        if not left_0hz_data.empty and not right_0hz_data.empty:
            t_stat_0hz, p_value_0hz = ttest_ind(left_0hz_data, right_0hz_data)
            y_to_annotate_0hz = max_val + 0.05 * max_val
            annotate_p_value(plt.gca(), x_positions[0], x_positions[1], y_to_annotate_0hz, p_value_0hz)
            local_p_values['0HZ'] = p_value_0hz  # Adding the p-value to the local dictionary
        local_p_values['0HZ'] = p_value_0hz
        if p_value_0hz < 0.05:  # Statistically significant
            colors_dict['Left']['0HZ'] = 'blue'
            colors_dict['Right']['0HZ'] = '#BB2525'
        else:
            colors_dict['Left']['0HZ'] = 'lightblue'
            colors_dict['Right']['0HZ'] = 'lightpink'

    if '50HZ' in sub_df_all['Stimulation'].values:
        # Comparing Left vs. Right Hemisphere for 50Hz
        left_50hz_data = sub_df_all[(sub_df_all['Stimulation'] == '50HZ') & (sub_df_all['Hemisphere'] == 'Left')]['Avg_Rotations_per_20_seconds']
        right_50hz_data = sub_df_all[(sub_df_all['Stimulation'] == '50HZ') & (sub_df_all['Hemisphere'] == 'Right')]['Avg_Rotations_per_20_seconds']

        if not left_50hz_data.empty and not right_50hz_data.empty:
            t_stat_50hz, p_value_50hz = ttest_ind(left_50hz_data, right_50hz_data)
            y_to_annotate_50hz = max_val + 0.15 * max_val
            annotate_p_value(plt.gca(), x_positions[-3], x_positions[-2], y_to_annotate_50hz, p_value_50hz)
            local_p_values['50HZ'] = p_value_50hz  # Adding the p-value to the local dictionary

    local_p_values['50HZ'] = p_value_50hz
    if p_value_50hz < 0.05:  # Statistically significant
        colors_dict['Left']['50HZ'] = 'blue'
        colors_dict['Right']['50HZ'] = '#BB2525'
    else:
        colors_dict['Left']['50HZ'] = 'lightblue'
        colors_dict['Right']['50HZ'] = 'lightpink'

    # Store the local p-values in the main dictionary
    if local_p_values:
        p_values_dict[brain_region] = local_p_values

    # Adjust the bar colors based on statistical significance
    for idx, label in enumerate(labels):
        stim = label.split(" ")[-1].replace("(", "").replace(")", "")
        if "Left" in label:
            bars[idx].set_color(colors_dict['Left'][stim])
        elif "Right" in label:
            bars[idx].set_color(colors_dict['Right'][stim])
    # ----- 

    # plt.xlabel("Stimulation and Hemisphere", fontsize=14)
    plt.xticks(x_positions, labels, rotation=0, fontsize=12)
    plt.ylabel("Average Clockwise Rotation", fontsize=14)
    plt.title(f"{brain_region}: Average Rotations per 20 Seconds", fontsize=14)
    plt.ylim(-3, 3)

    if len(bars) > 2:
        plt.legend([bars[0], bars[1], bars[-1]], ['Left Hemisphere', 'Right Hemisphere', 'Control'], fontsize=12)
    else:
        print(f"Not enough bars to add legends for {brain_region}")

    plt.text(0.02, 0.95, f"Total mice: {num_mice}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.90, f"Total sessions: {num_sessions}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.85, f"Total controls: {num_controls}", transform=plt.gca().transAxes)
    print(f'{brain_region}: \nN mice: {num_mice} \nN sessions: {num_sessions} \nN controls: {num_controls}')

    # Save the figure
    plt.savefig(f"/Users/feiyang/Projects/GLM-HMM/{brain_region}_average_rotations.png", dpi=300)

    plt.show()

# Convert the nested dictionary to a DataFrame
p_values_df = pd.DataFrame(p_values_dict).T

# Save the DataFrame to a CSV
# p_values_df.to_csv("/Users/Carolina/SLEAP_project/figs/p_values.csv")

print(p_values_df)