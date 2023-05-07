# import warnings to supress warning caused by bug in catboost 1.0.6
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use('TkAgg') #QtAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# For bag data
import bagpy
from bagpy import bagreader

import numpy as np
import os
import pandas as pd
pd.pandas.set_option('display.max_columns', None) # Show all columns of Dataframes
import scipy.constants as sc
import math
from math import sin
from math import cos
from math import radians
from math import atan2
from math import sqrt

# Constants
lat = 52.239 # latitude of citadel where data was recorded

# Path were the .txt files are saved
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
bd = str(desktop)+'/Group-9/'

# Going to that directory
os.chdir(bd)

# Listing all the files in the directory (30 txt files)
entries = os.listdir(bd)

# From Bag Files to CSV (done once)
# SOS SOS RUN ONLY ONCE
# for i in range(len(entries)):
#     group_member = bagreader(entries[i])
#     csvfiles = []
#     for t in group_member.topics:
#         data = group_member.message_by_topic(t)
#         csvfiles.append(data)
#     print("created csv files for bag number " + str(i+1))
# print("transformed all bag files into csv files")
# SOS SOS RUN ABOVE ONLY ONCE


# Preprocessing
# Remove outliers from data by using the Z-Score method
def detect_and_remove_outliers(df, threshold=2.5):
    # Create a copy of the input dataframe
    cleaned_df = df.copy()
    # Iterate over the columns of the dataframe
    for col in df.columns[1:]:
        # Calculate the mean and standard deviation of the column
        mean = np.mean(df[col])
        std = np.std(df[col])
        # Calculate the Z-scores of the column
        z_scores = (df[col] - mean) / std
        # Identify the data points with a Z-score above the threshold
        outliers = np.abs(z_scores) > threshold
        # Remove the outliers from the column
        cleaned_df[col] = df[col][~outliers]
    return cleaned_df

def transform_to_global_frame(x, yi, z, pitch, roll, yaw):
    # Convert pitch, roll, and yaw angles to radians
    p = math.radians(pitch)
    r = math.radians(roll)
    y = math.radians(yaw)

    # Create rotation matrix
    R = np.array([[(cos(y) * cos(r) - sin(y) * sin(p) * sin(r)), (-sin(y) * cos(p)), (cos(y) * sin(r) + sin(y) * sin(p) * cos(r))],
     [(sin(y) * cos(r) + cos(y) * sin(p) * sin(r)), (cos(y) * cos(p)), (sin(y) * sin(r) - cos(y) * sin(p) * cos(r))],
     [(-cos(p) * sin(r)), (sin(p)), (cos(p) * cos(r))]])
    # Transform data from body frame to global frame
    x_global = R[0][0] * x + R[0][1] * yi + R[0][2] * z
    y_global = R[1][0] * x + R[1][1] * yi + R[1][2] * z
    z_global = R[2][0] * x + R[2][1] * yi + R[2][2] * z

    return x_global, y_global, z_global

entries = os.listdir(bd)
# Make a list of the folder names by filtering out the entries that end in '.bag'
folder_names = [entry for entry in entries if not entry.endswith('.bag')]

# Initialize an empty list to store the dataframes
dfs = []

# Loop through the folder names
for folder in folder_names:
    # Construct the file path to the csv file inside the folder
    file_path = os.path.join(folder, 'imu-data.csv')

    # Read the csv file into a dataframe using pandas
    df = pd.read_csv(file_path, usecols=[0, 18, 19, 20, 30, 31, 32])

    # Add the dataframe to the list
    dfs.append(df)

# Bias and scale factor values selected:
bias_acc = np.array([0.002186, 0.010954, -0.012225])
scale_acc = np.array([0.000466, -0.000398, 0.000444])

bias_gyro = np.array([-0.003768, 0.000648, -0.001457])
scale_gyro = np.array([-1.000354, -0.990428, -0.982728])

# Initialize variables for trajectory estimation
time = []
velocity = []
displacement = []
initial_pitch = []
initial_roll = []
initial_azimuth = []
initial_Rbl = []
# Calibrate the IMU using the bias and scale factor values
for i in range(len(dfs)):
    #print(dfs[i].head())
    dfs[i]["linear_acceleration.x"] = ((dfs[i]["linear_acceleration.x"])-bias_acc[0])/(scale_acc[0]+1)
    dfs[i]["linear_acceleration.y"] = ((dfs[i]["linear_acceleration.y"]) - bias_acc[1]) / (scale_acc[1] + 1)
    dfs[i]["linear_acceleration.z"] = ((dfs[i]["linear_acceleration.z"]) - bias_acc[2]) / (scale_acc[2] + 1)

    dfs[i]["angular_velocity.x"] = ((dfs[i]["angular_velocity.x"]) - bias_gyro[0]) / (scale_gyro[0] + 1) # SOS SOS SOS MAYBE CHANGE HERE AS SCALE_GYRO IS VERY BAD SOS
    dfs[i]["angular_velocity.y"] = ((dfs[i]["angular_velocity.y"]) - bias_gyro[1]) / (scale_gyro[1] + 1)  # SOS SOS SOS MAYBE CHANGE HERE AS SCALE_GYRO IS VERY BAD SOS
    dfs[i]["angular_velocity.z"] = ((dfs[i]["angular_velocity.z"]) - bias_gyro[2]) / (scale_gyro[2] + 1)  # SOS SOS SOS MAYBE CHANGE HERE AS SCALE_GYRO IS VERY BAD SOS
    #print(dfs[i].head())

    dfs[i]["Pitch (rad)"] = np.arctan2(dfs[i]["linear_acceleration.y"], ((dfs[i]["linear_acceleration.x"])**2+(dfs[i]["linear_acceleration.z"])**2)**(1/2))
    dfs[i]["Roll (rad)"] = np.arctan2(-(dfs[i]["linear_acceleration.x"]), (dfs[i]["linear_acceleration.z"]))
    dfs[i]["Azimuth (rad)"] = np.arctan2((dfs[i]["angular_velocity.x"])*np.cos(dfs[i]["Roll (rad)"])+(dfs[i]["angular_velocity.z"])*np.sin(dfs[i]["Roll (rad)"]), (dfs[i]["angular_velocity.y"])*np.cos(dfs[i]["Pitch (rad)"])+(dfs[i]["angular_velocity.x"])*np.sin(dfs[i]["Roll (rad)"])*np.sin(dfs[i]["Pitch (rad)"])-(dfs[i]["angular_velocity.z"])*np.cos(dfs[i]["Roll (rad)"])*np.sin(dfs[i]["Pitch (rad)"]))

    # initial_pitch.append(atan2(dfs[i]["linear_acceleration.y"].iloc[0], sqrt((dfs[i]["linear_acceleration.x"].iloc[0])**2+(dfs[i]["linear_acceleration.z"].iloc[0])**2))) # In radians
    # initial_roll.append(atan2(-(dfs[i]["linear_acceleration.x"].iloc[0]), (dfs[i]["linear_acceleration.z"].iloc[0])))
    # initial_azimuth.append(atan2((dfs[i]["angular_velocity.x"].iloc[0])*cos(initial_roll[i])+(dfs[i]["angular_velocity.z"].iloc[0])*sin(initial_roll[i]), (dfs[i]["angular_velocity.y"].iloc[0])*cos(initial_pitch[i])+(dfs[i]["angular_velocity.x"].iloc[0])*sin(initial_roll[i])*sin(initial_pitch[i])-(dfs[i]["angular_velocity.z"].iloc[0])*cos(initial_roll[i])*sin(initial_pitch[i])))

    y = dfs[i]["Azimuth (rad)"].iloc[0]
    r = dfs[i]["Roll (rad)"].iloc[0]
    p = dfs[i]["Pitch (rad)"].iloc[0]

    initial_Rbl.append(np.array([[(cos(y) * cos(r) - sin(y) * sin(p) * sin(r)), (-sin(y) * cos(p)), (cos(y) * sin(r) + sin(y) * sin(p) * cos(r))],
     [(sin(y) * cos(r) + cos(y) * sin(p) * sin(r)), (cos(y) * cos(p)), (sin(y) * sin(r) - cos(y) * sin(p) * cos(r))],
     [(-cos(p) * sin(r)), (sin(p)), (cos(p) * cos(r))]]))


    ## Snipet code to check whether equation page 165 stands
    # print(initial_azimuth[i])
    # print("\n")
    # print(atan2(dfs[i]["angular_velocity.x"].iloc[0], dfs[i]["angular_velocity.y"].iloc[0]))
    # Extract the time values from the dataframe
    t = dfs[i]['Time'].values
    time.append(t)

    # Initialize variables for velocity and displacement estimation
    v = np.zeros((len(t), 3))
    d = np.zeros((len(t), 3))

    # Integrate the IMU data to estimate the velocity and displacement
    for j in range(1, len(t)):
        dt = t[j] - t[j-1]

        # Transform accelerometer data from the body frame to the global frame
        dfs[i]['linear_acceleration.x'].iloc[j], dfs[i]['linear_acceleration.y'].iloc[j], dfs[i]['linear_acceleration.z'].iloc[j] = transform_to_global_frame(dfs[i]['linear_acceleration.x'].iloc[j], dfs[i]['linear_acceleration.y'].iloc[j], dfs[i]['linear_acceleration.z'].iloc[j], p, r, y)

        # Transform gyroscope data from the body frame to the global frame
        dfs[i]['angular_velocity.x'], dfs[i]['angular_velocity.y'], dfs[i]['angular_velocity.z'] = transform_to_global_frame(dfs[i]['angular_velocity.x'], dfs[i]['angular_velocity.y'], dfs[i]['angular_velocity.z'], p, r, y)

        # Update pitch, roll, and yaw angles over time
        dfs[i]["Pitch (rad)"].iloc[j] = dfs[i]["Pitch (rad)"].iloc[j-1] + dfs[i]["angular_velocity.x"].iloc[j]*dt
        dfs[i]["Roll (rad)"].iloc[j] = dfs[i]["Roll (rad)"].iloc[j - 1] + dfs[i]["angular_velocity.y"].iloc[j] * dt
        dfs[i]["Azimuth (rad)"].iloc[j] = dfs[i]["Azimuth (rad)"].iloc[j - 1] + dfs[i]["angular_velocity.z"].iloc[j] * dt
        y = dfs[i]["Azimuth (rad)"].iloc[j]
        r = dfs[i]["Roll (rad)"].iloc[j]
        p = dfs[i]["Pitch (rad)"].iloc[j]

        acceleration = dfs[i][['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']].iloc[j].values
        acceleration = acceleration - np.array([0*-9.81*cos(p)*sin(r), 0*9.81*sin(p), 9.81]) # SOS we assume that the z-axis is aligned with the local gravitational field, maybe not the case
        v[j] = v[j-1] + acceleration * dt
        d[j] = d[j-1] + v[j] * dt
        # if (j%1000) == 0:
        #     plt.ion()
        #     plt.show()
        #     plt.scatter(d[:, 0], d[:, 1])
        #     plt.draw()
        #     plt.pause(0.000000000001)
        #     plt.show()

    # Append the velocity and displacement estimates to the list
    velocity.append(v)
    displacement.append(d)


for i in range(len(dfs)):
    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(displacement[i][:, 0], displacement[i][:, 1], displacement[i][:, 2])
    plt.show()

print("breakpoint")
# Apply preprocessing step to each dataframe
pro_dfs = []
for i in range(len(dfs)):
    # Remove the outliers from the data
    df = detect_and_remove_outliers(dfs[i])

    # Add the dataframe to the list
    pro_dfs.append(df)
