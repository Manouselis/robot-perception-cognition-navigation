# Script created by Panteleimon Manouselis M-ROB UTWENTE
# import warnings to supress warning caused by bug in catboost 1.0.6
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import scipy.constants as sc
from math import sin
from math import radians

# Constants
lat = 52.239 # latitude of citadel where data was recorded

# Path were the .txt files are saved
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
bd = str(desktop)+'/RPCN imu calibration txt/'

# Going to that directory
os.chdir(bd)

# Listing all the files in the directory (30 txt files)
entries = os.listdir(bd)


# Preprocessing
# Remove outliers from data by using the Z-Score method
def detect_and_remove_outliers(data, threshold=2.5):
  # Calculate the mean and standard deviation of the data
  mean = np.mean(data)
  std = np.std(data)
  # Calculate the Z-scores of the data
  z_scores = (data - mean) / std
  # Identify the data points with a Z-score above the threshold
  outliers = np.abs(z_scores) > threshold
  # Remove the outliers from the data
  cleaned_data = data[~outliers]
  return cleaned_data


# For accelerometer
bias_error_list_acc = []
scale_error_list_acc = []

# For gyro
bias_error_list_gyro = []
scale_error_list_gyro = []
for i in range(0, 30, 6):
    # Axis 1
    # txt to array
    df1 = pd.read_csv(entries[i], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])

    # Visualize the data before removing the outliers
    plt.figure()
    plt.plot(df1["Acc_Z"], '.')
    plt.title('Accelerometer Data before removing outliers (run number ' +str(i%5 +1)+')')

    # Remove the outliers from the data
    df1["Acc_Z"] = detect_and_remove_outliers(df1["Acc_Z"])

    # Visualize the data after removing the outliers
    plt.figure()
    plt.plot(df1["Acc_Z"], '.')
    plt.title('Accelerometer Data after removing outliers (run number ' +str(i%5 +1)+')')

    mean_Acc_Z = df1["Acc_Z"].mean() # F_up_Z

    # Second file
    df2 = pd.read_csv(entries[i+1], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df2["Acc_Z"] = detect_and_remove_outliers(df2["Acc_Z"])
    mean_Acc_Z_2 = df2["Acc_Z"].mean() # F_down_Z

    # Calculating Systematic Bias offset
    bias_error_Z = (mean_Acc_Z + mean_Acc_Z_2)/2

    # Calculating Systematic Scale Factor Error
    scale_error_Z = (mean_Acc_Z - mean_Acc_Z_2 - 2*sc.g)/(2*sc.g)


    # Axis 2
    df1 = pd.read_csv(entries[i+2], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df1["Acc_Y"] = detect_and_remove_outliers(df1["Acc_Y"])
    mean_Acc_Y = df1["Acc_Y"].mean() # F_up_Z

    df2 = pd.read_csv(entries[i+3], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df2["Acc_Y"] = detect_and_remove_outliers(df2["Acc_Y"])
    mean_Acc_Y_2 = df2["Acc_Y"].mean() # F_down_Z

    # Calculating Systematic Bias offset
    bias_error_Y = (mean_Acc_Y + mean_Acc_Y_2)/2

    # Calculating Systematic Scale Factor Error
    scale_error_Y = (mean_Acc_Y - mean_Acc_Y_2 - 2*sc.g)/(2*sc.g)


    # Axis 3
    df1 = pd.read_csv(entries[i+4], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df1["Acc_X"] = detect_and_remove_outliers(df1["Acc_X"])
    mean_Acc_X = df1["Acc_X"].mean() # F_up_Z

    df2 = pd.read_csv(entries[i+5], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df2["Acc_X"] = detect_and_remove_outliers(df2["Acc_X"])
    mean_Acc_X_2 = df2["Acc_X"].mean() # F_down_Z

    # Calculating Systematic Bias offset
    bias_error_X = (mean_Acc_X + mean_Acc_X_2)/2

    # Calculating Systematic Scale Factor Error
    scale_error_X = (mean_Acc_X - mean_Acc_X_2 - 2*sc.g)/(2*sc.g)

    # Adding erros in vectors
    bias_error_acc = pd.Series(data=[bias_error_X, bias_error_Y, bias_error_Z], index=["X", "Y", "Z"])
    bias_error_list_acc.append(bias_error_acc)
    scale_error_acc = pd.Series(data=[scale_error_X, scale_error_Y, scale_error_Z], index=["X", "Y", "Z"])
    scale_error_list_acc.append(scale_error_acc)



    # For gyro
    # Axis 1
    df1 = pd.read_csv(entries[i], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df1["Gyr_Z"] = detect_and_remove_outliers(df1["Gyr_Z"])
    mean_Gyr_Z = df1["Gyr_Z"].mean() # F_up_Z

    df2 = pd.read_csv(entries[i+1], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df2["Gyr_Z"] =detect_and_remove_outliers(df2["Gyr_Z"])
    mean_Gyr_Z_2 = df2["Gyr_Z"].mean() # F_down_Z

    # Calculating Systematic Bias offset
    bias_error_Z_gyr = (mean_Gyr_Z + mean_Gyr_Z_2)/2

    # Calculating Systematic Scale Factor Error
    scale_error_Z_gyr = (mean_Gyr_Z - mean_Gyr_Z_2 - 2*(15.04/3600)*sin(radians(lat)))/(2*(15.04/3600)*sin(radians(lat)))


    # Axis 2
    df1 = pd.read_csv(entries[i+2], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df1["Gyr_Y"] = detect_and_remove_outliers(df1["Gyr_Y"])
    mean_Gyr_Y = df1["Gyr_Y"].mean() # F_up_Z

    df2 = pd.read_csv(entries[i+3], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df2["Gyr_Y"] = detect_and_remove_outliers(df2["Gyr_Y"])
    mean_Gyr_Y_2 = df2["Gyr_Y"].mean() # F_down_Z

    # Calculating Systematic Bias offset
    bias_error_Y_gyr = (mean_Gyr_Y + mean_Gyr_Y_2)/2

    # Calculating Systematic Scale Factor Error
    scale_error_Y_gyr = (mean_Gyr_Y - mean_Gyr_Y_2 - 2*(15.04/3600)*sin(radians(lat)))/(2*(15.04/3600)*sin(radians(lat)))


    # Axis 3
    df1 = pd.read_csv(entries[i+4], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df1["Gyr_X"] = detect_and_remove_outliers(df1["Gyr_X"])
    mean_Gyr_X = df1["Gyr_X"].mean() # F_up_Z

    df2 = pd.read_csv(entries[i+5], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
    df2["Gyr_X"] = detect_and_remove_outliers(df2["Gyr_X"])
    mean_Gyr_X_2 = df2["Gyr_X"].mean() # F_down_Z

    # Calculating Systematic Bias offset
    bias_error_X_gyr = (mean_Gyr_X + mean_Gyr_X_2)/2

    # Calculating Systematic Scale Factor Error
    scale_error_X_gyr = (mean_Gyr_X - mean_Gyr_X_2 - 2*(15.04/3600)*sin(radians(lat)))/(2*(15.04/3600)*sin(radians(lat)))

    # Adding erros in vectors
    bias_error_gyr = pd.Series(data=[bias_error_X_gyr, bias_error_Y_gyr, bias_error_Z_gyr], index=["X", "Y", "Z"])
    bias_error_list_gyro.append(bias_error_gyr)
    scale_error_gyr = pd.Series(data=[scale_error_X_gyr, scale_error_Y_gyr, scale_error_Z_gyr], index=["X", "Y", "Z"])
    scale_error_list_gyro.append(scale_error_gyr)


# Create a DataFrame from the lists
print("bias error Accelerometer")
print(bias_error_list_acc)
print("\n")

print("scale factor error Accelerometer")
print(scale_error_list_acc)
print("\n")

print("bias error Gyroscope")
print(bias_error_list_gyro)
print("\n")

print("scale factor error Gyroscope")
print(scale_error_list_gyro)
print("\n")

# Run-to-run bias offset
run_to_run_bias_acc = pd.DataFrame(columns=["X", "Y", "Z"], index=[1, 2, 3, 4])
run_to_run_bias_gyro = pd.DataFrame(columns=["X", "Y", "Z"], index=[1, 2, 3, 4])

# Run-to-run scale factor (Scale factor instability)
Scale_factor_instability_acc = pd.DataFrame(columns=["X", "Y", "Z"], index=[1, 2, 3, 4])
Scale_factor_instability_gyro = pd.DataFrame(columns=["X", "Y", "Z"], index=[1, 2, 3, 4])

for i in range(len(bias_error_list_acc) - 1):
    # X - axis
    run_to_run_bias_acc.iloc[i, 0] = bias_error_list_acc[i + 1][0] - bias_error_list_acc[i][0]
    run_to_run_bias_gyro.iloc[i, 0] = bias_error_list_gyro[i+1][0]-bias_error_list_gyro[i][0]

    Scale_factor_instability_acc.iloc[i, 0] = scale_error_list_acc[i + 1][0] - scale_error_list_acc[i][0]
    Scale_factor_instability_gyro.iloc[i, 0] = scale_error_list_gyro[i+1][0]-scale_error_list_gyro[i][0]

    # Y - axis
    run_to_run_bias_acc.iloc[i, 1] = bias_error_list_acc[i + 1][1] - bias_error_list_acc[i][1]
    run_to_run_bias_gyro.iloc[i, 1] = bias_error_list_gyro[i + 1][1] - bias_error_list_gyro[i][1]

    Scale_factor_instability_acc.iloc[i, 1] = scale_error_list_acc[i + 1][1] - scale_error_list_acc[i][1]
    Scale_factor_instability_gyro.iloc[i, 1] = scale_error_list_gyro[i + 1][1] - scale_error_list_gyro[i][1]

    # Z - axis
    run_to_run_bias_acc.iloc[i, 2] = bias_error_list_acc[i + 1][2] - bias_error_list_acc[i][2]
    run_to_run_bias_gyro.iloc[i, 2] = bias_error_list_gyro[i + 1][2] - bias_error_list_gyro[i][2]

    Scale_factor_instability_acc.iloc[i, 2] = scale_error_list_acc[i + 1][2] - scale_error_list_acc[i][2]
    Scale_factor_instability_gyro.iloc[i, 2] = scale_error_list_gyro[i + 1][2] - scale_error_list_gyro[i][2]

# We only use data from runs (2, 3 and 4)
run_to_run_bias_acc = run_to_run_bias_acc.iloc[[1,2]]
run_to_run_bias_gyro = run_to_run_bias_gyro.iloc[[1,2]]

Scale_factor_instability_acc = Scale_factor_instability_acc.iloc[[1,2]]
Scale_factor_instability_gyro = Scale_factor_instability_gyro.iloc[[1,2]]

print("run to run bias error Accelerometer")
print(run_to_run_bias_acc)
print("\n\n")

print("run to run bias error Gyroscope")
print(run_to_run_bias_gyro)
print("\n\n")

print("Scale factor instability Accelerometer")
print(Scale_factor_instability_acc)
print("\n\n")

print("Scale factor instability Gyroscope")
print(Scale_factor_instability_gyro)
print("\n\n")


# Define a function to calculate the non-orthogonality matrix of a gyroscope
def calc_non_orthogonality_matrix(measurements):
  # Extract the measured outputs for each axis
  x = measurements[0]
  y = measurements[1]
  z = measurements[2]

  # Calculate the non-orthogonality matrix using the formula above
  N = [[(x**2 + y**2 + z**2 - x*y - x*z - y*z) / 2, (y*z - x**2) / 2, (x*z - y**2) / 2],
       [(y*z - x**2) / 2, (x**2 + y**2 + z**2 - x*y - x*z - y*z) / 2, (x*y - z**2) / 2],
       [(x*z - y**2) / 2, (x*y - z**2) / 2, (x**2 + y**2 + z**2 - x*y - x*z - y*z) / 2]]

  return N

plt.show()

x = 0
while x < 1 or x > 5:
    print("For what run of the experiment (1-5) should I calculate the calibration parameters?")
    x = input()
    x = int(x)

i = (x-1)*6
df1 = pd.read_csv(entries[i], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
gyr_z_mean = df1["Gyr_Z"].mean()  # F_up_Z
acc_z_mean = df1["Acc_Z"].mean()

df2 = pd.read_csv(entries[i + 2], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
gyr_y_mean = df2["Gyr_Y"].mean() # F_up_Y
acc_y_mean = df2["Acc_Y"].mean()

df3 = pd.read_csv(entries[i+4], sep="	", header=0, skiprows=12, usecols=[0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16])
gyr_x_mean = df3["Gyr_X"].mean()  # F_up_X
acc_x_mean = df3["Acc_X"].mean()

measurements_gyro = [gyr_x_mean, gyr_y_mean, gyr_z_mean]
non_orthogonality_matrix_gyr = calc_non_orthogonality_matrix(measurements_gyro)


measurements_acc = [acc_x_mean, acc_y_mean, acc_z_mean]
non_orthogonality_matrix_acc = calc_non_orthogonality_matrix(measurements_acc)

# Final calibration parameters
# Gyro
print("\nCalibration parameters for gyroscope should be:")
print("Instrument Bias Error (used run number " + str(x) + ")")
print(bias_error_list_gyro[x-1])
print("\nGyroscope Scale Error (used run number " + str(x) + ")")
print(scale_error_list_gyro[x-1])
print("\nNon-orthogonality matrix of gyroscope (used run number " + str(x) + ")")
print(non_orthogonality_matrix_gyr)

# Accelerometer
print("\nCalibration parameters for accelerometer should be:")
print("Instrument Bias Error (used run number " + str(x) + ")")
print(bias_error_list_acc[1])
print("\nAccelerometer Scale Error (used run number " + str(x) + ")")
print(scale_error_list_acc[1])
print("\nNon-orthogonality matrix of accelerometer (used run number " + str(x) + ")")
print(non_orthogonality_matrix_acc)
