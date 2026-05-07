import streamlit as st
import fastf1 as ff1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor 
import numpy as np

st.set_page_config(page_title="F1 Advanced Strategy Pro", layout="wide")
st.title("🏎️ F1 Advanced Strategy Optimizer")

ff1.Cache.enable_cache('cache_folder')

st.sidebar.header("Race Settings")
year = st.sidebar.selectbox("Year", [2024, 2025])
track = st.sidebar.text_input("Track Name", "Silverstone")

st.sidebar.subheader("Head-to-Head")
driver1 = st.sidebar.selectbox("Driver 1", ["HAM", "VER", "NOR", "LEC", "PIA"], index=0)
driver2 = st.sidebar.selectbox("Driver 2", ["HAM", "VER", "NOR", "LEC", "PIA"], index=1)

@st.cache_data
def load_f1_data(year, track, driver_code):
    session = ff1.get_session(year, track, 'R')
    session.load()
    laps = session.laps.pick_driver(driver_code).dropna(subset=['LapTime', 'TyreLife'])
    X = laps[['TyreLife']].values
    y = laps['LapTime'].dt.total_seconds().values
    return X, y, session.weather_data

def train_advanced_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

X1, y1, weather1 = load_f1_data(year, track, driver1)
X2, y2, weather2 = load_f1_data(year, track, driver2)

model1 = train_advanced_model(X1, y1)
model2 = train_advanced_model(X2, y2)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📊 {driver1} Analysis")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X1, y1, color='cyan', alpha=0.4, label='Actual')
    line_X = np.linspace(X1.min(), X1.max(), 100).reshape(-1, 1)
    ax1.plot(line_X, model1.predict(line_X), color='blue', label='RF Prediction')
    st.pyplot(fig1)

with col2:
    st.subheader(f"📊 {driver2} Analysis")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X2, y2, color='orange', alpha=0.4, label='Actual')
    line_X2 = np.linspace(X2.min(), X2.max(), 100).reshape(-1, 1)
    ax2.plot(line_X2, model2.predict(line_X2), color='red', label='RF Prediction')
    st.pyplot(fig2)

st.divider()
st.subheader("🤖 AI Strategy Suggestion")
test_lap = st.slider("Simulate Tire Age (Laps)", 1, 40, 15)

p1 = model1.predict([[test_lap]])[0]
p2 = model2.predict([[test_lap]])[0]

if p1 < p2:
    st.info(f"Strategy Tip: **{driver1}** is faster than **{driver2}** on {test_lap} lap old tires by {(p2-p1):.2f}s!")
else:
    st.info(f"Strategy Tip: **{driver2}** is gaining an advantage of {(p1-p2):.2f}s over **{driver1}**!")


@st.cache_data
def load_advanced_data(year, track, driver_code):
    session = ff1.get_session(year, track, 'R')
    session.load()
    laps = session.laps.pick_driver(driver_code).dropna(subset=['LapTime', 'TyreLife'])
    
    weather = session.weather_data
    track_temp = weather['TrackTemp'].mean() 
    
    X = laps[['TyreLife']].values
    y = laps['LapTime'].dt.total_seconds().values
    return X, y, track_temp

X1, y1, temp1 = load_advanced_data(year, track, driver_code=driver1)
X2, y2, temp2 = load_advanced_data(year, track, driver2)

st.sidebar.markdown(f"### 🌡️ Live Track Conditions")
st.sidebar.metric("Avg Track Temp", f"{temp1:.1f}°C")

st.divider()
st.header("🤖 AI Strategy Optimizer")

target_lap = st.slider("Select Lap for Prediction", 1, 52, 25)

p1 = train_advanced_model(X1, y1).predict([[target_lap]])[0]
p2 = train_advanced_model(X2, y2).predict([[target_lap]])[0]

col_a, col_b = st.columns(2)
col_a.metric(f"{driver1} Predicted Pace", f"{p1:.2f}s")
col_b.metric(f"{driver2} Predicted Pace", f"{p2:.2f}s")

if temp1 > 40:
    st.warning("⚠️ **Extreme Track Temp Alert:** Tires will degrade 20% faster than normal. Consider an early pit-stop!")

if abs(p1 - p2) < 0.2:
    st.success(f"Strategy: Both drivers are on similar pace. Overtake possible only via Under-cut!")
elif p1 < p2:
    st.info(f"Strategy: **{driver1}** has the pace advantage. **{driver2}** should defend or box now!")
else:
    st.info(f"Strategy: **{driver2}** is charging! **{driver1}** is losing the tire window.")    
st.divider()
st.header("🏁 AI Pit-Stop Optimizer")

def find_pit_window(model, max_laps=50):
    paces = model.predict(np.arange(1, max_laps).reshape(-1, 1))
    cliff_lap = np.where(paces > paces[0] + 1.5)[0]
    if len(cliff_lap) > 0:
        return cliff_lap[0]
    return 30 

win1 = find_pit_window(model1) 

win2 = find_pit_window(model2)

c1, c2 = st.columns(2)
with c1:
    st.info(f"**{driver1} Pit Window:** Laps {win1-2} to {win1+2}")
    st.write(f"Reason: AI detects a pace drop after lap {win1}.")
with c2:
    st.info(f"**{driver2} Pit Window:** Laps {win2-2} to {win2+2}")
    st.write(f"Reason: Optimal tire life ends around lap {win2}.")    
st.sidebar.markdown("---")
st.sidebar.subheader("Live Race Gap")

current_gap = p1 - p2 

if current_gap > 0:
    st.sidebar.warning(f"{driver2} is {abs(current_gap):.3f}s faster per lap!")
else:
    st.sidebar.success(f"{driver1} is {abs(current_gap):.3f}s faster per lap!")    

    