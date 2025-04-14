import pandas as pd
import numpy as np

# Define the number of lines to generate
num_lines = 5000

# Define the header
header = ['Timestamp', 'MachineID', 'Temperature', 'Vibration', 'Pressure', 'OperatingHours', 'MaintenanceFlag', 'QualityScore', 'ProductionRate', 'PowerConsumption', 'LubricationLevel', 'ErrorCodes', 'ProductQuality', 'UnitsProduced']

# Create an empty list to store the data
data = []

# Generate the data
for i in range(num_lines):
    timestamp = pd.Timestamp('2025-04-01 08:00:00') + pd.Timedelta(hours=i)
    machine_id = f'Machine_{i % 10 + 1:02d}'
    temperature = np.random.uniform(40, 55)
    vibration = np.random.uniform(0.8, 1.5)
    pressure = np.random.uniform(110, 140)
    operating_hours = np.random.randint(1000, 2000)
    maintenance_flag = np.random.choice([0, 1], p=[0.9, 0.1])
    quality_score = np.random.uniform(0.8, 1.0)
    production_rate = np.random.randint(30, 50)
    power_consumption = np.random.uniform(3.0, 4.5)
    lubrication_level = np.random.randint(60, 80)
    error_codes = np.random.choice([0, 101, 105], p=[0.7, 0.15, 0.15])
    product_quality = np.random.uniform(0.75, 0.95)
    units_produced = np.random.randint(70, 95)
    
    data.append([timestamp, machine_id, temperature, vibration, pressure, operating_hours, maintenance_flag, quality_score, production_rate, power_consumption, lubrication_level, error_codes, product_quality, units_produced])

# Create a Pandas DataFrame from the data
df = pd.DataFrame(data, columns=header)

# Save the DataFrame to a CSV file
csv_file_path = 'c:\\Users\\athar\\OneDrive\\Documents\\krisha project\\predictive\\data\\generated_machine_data.csv'
df.to_csv(csv_file_path, index=False)

print(f'Successfully generated CSV file with {num_lines} lines at {csv_file_path}')
