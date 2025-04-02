# 🏎️ F1 Race Time Prediction - 2025 Japanese GP

### 🚀 Predicting Race Lap Times with Machine Learning & Weather Data

## 📌 Overview
This project predicts **Formula 1 race lap times** for the 2025 Japanese Grand Prix 🏁 using **machine learning** and **real-time weather data** from OpenWeatherMap.

## 🔥 Features
✅ Uses **FastF1** to fetch real race data 📊  
✅ Incorporates **real-time weather conditions** ☀️🌧️  
✅ **Gradient Boosting Regressor** for prediction 🔥  
✅ **Scikit-learn StandardScaler** for feature scaling ⚖️  
✅ Supports multiple **F1 drivers** 👨‍🔧🏎️  

---

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Jaswanth-aditya/f1-race-prediction.git
cd f1-race-prediction
```
### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 3️⃣ Set Your API Key (OpenWeatherMap)
Replace the `API_KEY` in `predictions-japanese_gp.py` with **your own** OpenWeatherMap API key.

### 4️⃣ Run the Prediction Script
```sh
python predictions-japanese_gp.py
```

---

## 🧠 How It Works
1. **Fetches historical F1 data** from FastF1 for 2024.
2. **Calculates average lap & sector times** for each driver.
3. **Fetches real-time weather data** (Temperature, Wind Speed, Rain).
4. **Trains a Gradient Boosting Regressor** using qualifying & sector times.
5. **Predicts 2025 race lap times** using the trained model.

---

## 📊 Example Output
```
🏁 2025 Japanese GP Predictions 🌞
---------------------------------
1. Max Verstappen - 90.32s
2. Charles Leclerc - 90.65s
3. Lewis Hamilton - 90.89s
...

Weather Impact: Dry Conditions ☀️
Temperature: 22°C | Humidity: 58% | Wind: 3.5 m/s
```

---

## 🤝 Contributing
Pull requests are welcome! Feel free to fork the repo and submit improvements. 🚀

---

## 📜 License
MIT License © 2025 Jaswanth Aditya

---

## 🌟 Show Some Love!
If you like this project, give it a ⭐ on GitHub! 😍🔥

