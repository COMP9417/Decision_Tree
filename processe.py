import numpy as np
import pandas as pd

num_rows = 23000

# Read the CSV file
file_name = 'mini_train.csv'  # Replace this with your CSV file name
data = pd.read_csv(file_name, nrows=num_rows)

# Drop the first three columns and 'site_id', 'precip_depth_1_hr', and 'primary_use' columns
data = data.drop(columns=['meter', 'timestamp', 'site_id', 'precip_depth_1_hr', 'primary_use'])

# Convert numeric columns to float type
data[['meter_reading', 'square_feet', 'year_built', 'floor_count', 'air_temperature',
      'cloud_coverage', 'dew_temperature', 'sea_level_pressure', 'wind_direction',
      'wind_speed']] = data[['meter_reading', 'square_feet', 'year_built', 'floor_count', 'air_temperature',
                             'cloud_coverage', 'dew_temperature', 'sea_level_pressure',
                             'wind_direction', 'wind_speed']].astype(float)

# Remove rows with 'meter_reading' equal to 0 and apply a logarithmic transformation to the 'meter_reading' column
data = data[data['meter_reading'] != 0]
data['meter_reading'] = np.log1p(data['meter_reading'])

# Replace NaN values with the mean value of each column
data = data.fillna(data.mean())

# Perform further analysis or modeling on the processed data
print(data.iloc[700])
print(data.dtypes)

# Save the processed data to a new CSV file
output_file_name = 'finalresult.csv'  # Replace this with the desired file name
data.to_csv(output_file_name, index=False)

