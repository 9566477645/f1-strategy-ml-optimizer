import fastf1 as ff1
from fastf1 import plotting
import matplotlib.pyplot as plt

plotting.setup_mpl()

session = ff1.get_session(2024, 'Silverstone', 'Q')
session.load()

ham_lap = session.laps.pick_driver('HAM').pick_fastest()
ver_lap = session.laps.pick_driver('VER').pick_fastest()

ham_telemetry = ham_lap.get_telemetry().add_distance()
ver_telemetry = ver_lap.get_telemetry().add_distance()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ham_telemetry['Distance'], ham_telemetry['Speed'], label='Hamilton', color='cyan')

ax.plot(ver_telemetry['Distance'], ver_telemetry['Speed'], label='Verstappen', color='red')

ax.set_xlabel('Distance in meters')
ax.set_ylabel('Speed in km/h')
ax.set_title('Hamilton vs Verstappen - Fastest Lap Speed Trace')
ax.legend()
plt.show()