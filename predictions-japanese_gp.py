import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import requests

# OpenWeatherMap API Configuration
API_KEY = ''   # Add you API key here.
CITY_NAME = 'Suzuka'
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

def get_weather_data(api_key, city_name):
    complete_url = f"{BASE_URL}q={city_name}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    data = response.json()
    if data["cod"] == "404":
        return None
    else:
        rain = 1 if 'rain' in [x['main'].lower() for x in data['weather']] else 0
        return {
            "Temperature": data["main"]["temp"],
            "Humidity": data["main"]["humidity"],
            "WindSpeed": data["wind"]["speed"],
            "Rain": rain
        }

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Japanese GP data
session_2024 = fastf1.get_session(2024, "Japan", "R")
session_2024.load()

# Get historical rain data from 2024 race
rain_2024 = 1 if session_2024.weather_data['Rainfall'].sum() > 0 else 0

# Extract and process lap data
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds and add rain feature
time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
for col in time_cols:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()
laps_2024["Rain"] = rain_2024

# Calculate average sector times for 2024 drivers
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "Rain"]].mean().reset_index()

# 2025 Qualifying Data
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.018, 92.092, 92.141, 92.174]
})

# Driver code mapping
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", 
    "Max Verstappen": "VER", "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC",
    "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT", "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI",
    "Pierre Gasly": "GAS", "Oliver Bearman": "BEA", "Jack Doohan": "DOO",
    "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

# Create DriverCode column
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge with 2024 sector times
merged_data = qualifying_2025.merge(
    sector_times_2024, 
    left_on="DriverCode", 
    right_on="Driver", 
    how="left"
)

# Get current weather data for 2025 prediction
current_weather = get_weather_data(API_KEY, CITY_NAME)
rain_2025 = current_weather["Rain"] if current_weather else 0

# Fill missing sector times with 2024 averages and add 2025 rain
sector_avg = sector_times_2024[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean()
for col in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
    merged_data[col] = merged_data[col].fillna(sector_avg[col])
merged_data["Rain"] = rain_2025  # Use current rain status for prediction

# Get 2024 lap times and fill missing values
lap_time_2024 = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = merged_data.merge(lap_time_2024, left_on="DriverCode", right_on="Driver", how="left")
merged_data["LapTime (s)"] = merged_data["LapTime (s)"].fillna(lap_time_2024["LapTime (s)"].mean())

# Prepare features and target
features = ["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "Rain"]
X = merged_data[features]
y = merged_data["LapTime (s)"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, random_state=42)
model.fit(X_scaled, y)

# Predict race times
X_pred = scaler.transform(merged_data[features])
qualifying_2025["PredictedRaceTime (s)"] = model.predict(X_pred)
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Display results
print("\n🏁 2025 Japanese GP Predictions with Rain Factor 🌧️" if rain_2025 else "\n🏁 2025 Japanese GP Predictions 🌞")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]].reset_index(drop=True))

# Display weather impact
if current_weather:
    rain_status = "Rain Expected ☔" if rain_2025 else "Dry Conditions ☀️"
    print(f"\nWeather Impact: {rain_status}")
    print(f"Temperature: {current_weather['Temperature']}°C | Humidity: {current_weather['Humidity']}% | Wind: {current_weather['WindSpeed']} m/s")