import fastf1 as ff1
import pandas as pd
import matplotlib.pyplot as plt

ff1.Cache.enable_cache('cache_folder') 

session = ff1.get_session(2024, 'Silverstone', 'R') 
session.load() 

all_laps = session.laps
ml_data = all_laps[['LapNumber', 'LapTime', 'Stint', 'TyreLife', 'Compound', 'Driver']].copy()

ml_data['LapTime_Seconds'] = ml_data['LapTime'].dt.total_seconds()
ml_data = ml_data.dropna()

print("--- Success! Data is Ready for ML ---")
print(ml_data.head())

plt.figure(figsize=(10, 6))
for driver in ['HAM', 'VER', 'NOR']:
    driver_data = ml_data[ml_data['Driver'] == driver]
    plt.scatter(driver_data['TyreLife'], driver_data['LapTime_Seconds'], label=driver)

plt.xlabel('Tire Age (Laps)')
plt.ylabel('Lap Time (Seconds)')
plt.title('Impact of Tire Wear - 2024 Silverstone')
plt.legend()
plt.grid(True)
plt.show()