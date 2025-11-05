# pv_ml_simple.py
# Zweck: aus GTI + Temperatur die PV-Leistung (P_W) vorhersagen
# Voraussetzungen: pandas, numpy, matplotlib, scikit-learn, requests

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# 1) Einstellungen / Datenzeitraum
# --------------------------
lat, lon = 50.94, 6.96   # Köln (anpassbar)
start_date, end_date = "2025-04-01", "2025-06-30"  # Beispiel: 3 Monate Training
timezone = "Europe/Berlin"

# --------------------------
# 2) Daten holen (Open-Meteo Archive)
# --------------------------
def fetch_open_meteo_archive(lat, lon, start_date, end_date, timezone="Europe/Berlin"):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=global_tilted_irradiance,temperature_2m"
        f"&timezone={timezone}"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame({
        "time": pd.to_datetime(j["hourly"]["time"]),
        "GTI": j["hourly"]["global_tilted_irradiance"],
        "T_air": j["hourly"]["temperature_2m"]
    })
    df = df.set_index("time").sort_index()
    return df

print("Lade Wetterdaten...")
df = fetch_open_meteo_archive(lat, lon, start_date, end_date, timezone)
print("Datenpunkte:", len(df))

# --------------------------
# 3) Physikalisches Target erzeugen (synthetische 'Messdaten')
# --------------------------
# Systemannahmen (klar dokumentieren!)
A = 10.0        # Modulfläche in m^2
eta0 = 0.18     # Nennwirkungsgrad bei 25°C
gamma = 0.004   # Temperaturkoeffizient (1/°C)
T_ref = 25.0

# vereinfachte Zelltemperatur-Näherung (nur für synthetisches Target)
cell_temp_coeff = 0.03  # skaliert; empirisch einfach gewählt
df["T_cell"] = df["T_air"] + df["GTI"] * cell_temp_coeff / 100.0
df["eta_eff"] = eta0 * (1 - gamma * (df["T_cell"] - T_ref))
df["P_W"] = (df["GTI"] * A * df["eta_eff"]).clip(lower=0).fillna(0)  # Momentanleistung [W]

# --------------------------
# 4) Feature Engineering
# --------------------------
df_feat = df[["GTI", "T_air", "P_W"]].copy()
df_feat["hour"] = df_feat.index.hour
# zyklische Kodierung der Stunde
df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24.0)
df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24.0)
# einfache Lag-Features (Nowcasting-Fähigkeit)
df_feat["GTI_lag1"] = df_feat["GTI"].shift(1).fillna(0)
df_feat["P_lag1"] = df_feat["P_W"].shift(1).fillna(0)
# evtl. rollende Mittelwerte
df_feat["GTI_roll3"] = df_feat["GTI"].rolling(window=3, min_periods=1).mean()

# Drop NaNs falls vorhanden
df_feat = df_feat.dropna()

# Feature- und Target-Matrizen
feature_cols = ["GTI", "T_air", "hour_sin", "hour_cos", "GTI_lag1", "P_lag1", "GTI_roll3"]
X = df_feat[feature_cols].values
y = df_feat["P_W"].values
times = df_feat.index

# --------------------------
# 5) Zeitbasierter Split (kein Zufallssplit)
# --------------------------
split_idx = int(len(df_feat) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
time_test = times[split_idx:]

# --------------------------
# 6) Modell (Pipeline) trainieren
# --------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

print("Trainiere RandomForest...")
pipeline.fit(X_train, y_train)

# --------------------------
# 7) Evaluation
# --------------------------
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Evaluation auf Testset: MAE={mae:.1f} W, RMSE={rmse:.1f} W, R2={r2:.3f}")

# Parity-Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, s=6, alpha=0.4)
mx = max(y_test.max(), y_pred.max())
plt.plot([0,mx],[0,mx],'r--')
plt.xlabel("True P [W]")
plt.ylabel("Predicted P [W]")
plt.title("Parity Plot")
plt.grid(True); plt.tight_layout(); plt.show()

# Zeitreihen-Plot (kleiner Abschnitt)
plt.figure(figsize=(12,4))
plt.plot(time_test, y_test, label="True", alpha=0.8)
plt.plot(time_test, y_pred, label="Predicted", alpha=0.7)
plt.legend(); plt.xlabel("Zeit"); plt.ylabel("Leistung [W]"); plt.title("Vorhersage vs. True (Test)")
plt.grid(True); plt.tight_layout(); plt.show()

# --------------------------
# 8) Tagesenergie (Trapezregel) und Vergleich für Testperiode
# --------------------------
df_comp = pd.DataFrame(index=time_test)
df_comp["true_W"] = y_test
df_comp["pred_W"] = y_pred

# Funktion: trapezregel pro Tag (stündliche Daten -> Wh)
def daily_energy_trapz(series):
    # series indexed by hour; ensure sorted
    y = series.values
    # x = [0,1,2,...] hours - wenn Daten stündlich ist das korrekt
    x = np.arange(len(y))
    return np.trapz(y, x)  # liefert Wh analog (W * h)

daily_true = df_comp["true_W"].resample("D").apply(daily_energy_trapz)
daily_pred = df_comp["pred_W"].resample("D").apply(daily_energy_trapz)

comparison = pd.DataFrame({"true_Wh": daily_true, "pred_Wh": daily_pred}).dropna()
print("Tagesenergie Beispiel (erste 10 Tage des Tests):")
print(comparison.head(10))

# --------------------------
# 9) Save model & results (optional)
# --------------------------
# Falls gewünscht: pickle export
import joblib
joblib.dump(pipeline, "rf_pv_model.joblib")
df_comp.to_csv("pv_predictions_test_period.csv")
print("Model gespeichert: rf_pv_model.joblib, Ergebnisse: pv_predictions_test_period.csv")
