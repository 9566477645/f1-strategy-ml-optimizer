import fastf1 as ff1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

ff1.Cache.enable_cache('cache_folder')
session = ff1.get_session(2024, 'Silverstone', 'R')
session.load()

laps = session.laps.pick_driver('HAM').dropna(subset=['LapTime', 'TyreLife'])
X = laps[['TyreLife']].values 
y = laps['LapTime'].dt.total_seconds().values


model = LinearRegression()
model.fit(X, y) 

future_tire_age = np.array([[35]])
predicted_time = model.predict(future_tire_age)

print(f"\n--- ML Prediction for HAM ---")
print(f"Predicted Lap Time at 35 laps tire age: {predicted_time[0]:.2f} seconds")

plt.scatter(X, y, color='blue', label='Actual Data') # Old data
plt.plot(X, model.predict(X), color='red', linewidth=3, label='ML Prediction Line') 
plt.ylabel('Lap Time (Seconds)')
plt.title('Phase 3: ML Predicting Lap Time Decay')
plt.legend()
plt.show()