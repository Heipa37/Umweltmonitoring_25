from senseboxAPI import SenseBox
import datetime as dt
from datetime import timezone
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import time
import pytz

def get_temperature_data(box_id: str, days: int = 30) -> pd.DataFrame:
    sb = SenseBox(box_id)
    sensor_infos = sb.get_sensor_info()
    temperature_sensor = next(s for s in sensor_infos if "Temperatur" in s["title"])
    if temperature_sensor is None:
        raise ValueError("Kein Sensor mit Titel 'Temperatur' gefunden.")

    sensor_id = temperature_sensor.get("sensor_id")
    if sensor_id is None:
        raise KeyError("Key 'sensorId' nicht im Sensorobjekt enthalten.")
    datetime_to = pd.to_datetime(temperature_sensor["lastMeasurement"])
    datetime_from = datetime_to - pd.Timedelta(days=days)
    
    df = sb.get_sensor_data(sensor_id=sensor_id, datetime_from=datetime_from, datetime_to=datetime_to)
    return df




def resample_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rundet den Zeitstempel auf die Minute herunter.
    """
    df = df.copy()
    df["measurement_time"] = pd.to_datetime(df["measurement_time"])
    df["measurement_time"] = df["measurement_time"].dt.floor("min")
    return df

def add_lag_features(df: pd.DataFrame, target_col: str = "measurement", lags: int = 5):
    """
    Fügt Lag-Features (z. B. t-1 bis t-5) aus der Zielspalte hinzu.

    Args:
        df (pd.DataFrame): Eingabedaten mit Zeitreihe.
        target_col (str): Name der Spalte, aus der Lags gebildet werden.
        lags (int): Anzahl der Lags, die erzeugt werden sollen.

    Returns:
        pd.DataFrame: DataFrame mit zusätzlichen Lag-Spalten.
    """
    df = df.copy()
    for i in range(1, lags + 1):
        df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)
    return df


def add_temporal_features(df: pd.DataFrame, time_col: str = "measurement_time") -> pd.DataFrame:
    """
    adds temporal Features:
    - day_of_year
    - is_day (6–18 Uhr)
    - is_weekend (Sa/So)
    - hour_sin, hour_cos
    - weekday_sin, weekday_cos
    - season_Winter, season_Spring, season_Summer, season_Autumn (One-Hot)
    """
    df = df.copy()

    # Zeitspalte oder Index vorbereiten
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        dt = df[time_col]
    else:
        df.index = pd.to_datetime(df.index)
        dt = df.index

    # Basiselemente
    df["day_of_year"] = dt.dt.dayofyear
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["is_weekend"] = dt.dt.weekday >= 5
    df["is_day"] = ((dt.dt.hour >= 6) & (dt.dt.hour < 18)).astype(int)

    # Zyklische Kodierung
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Jahreszeit bestimmen
    def get_season(dt_obj):
        m = dt_obj.month
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["season"] = dt.map(get_season)

    # One-Hot-Encoding
    df = pd.get_dummies(df, columns=["season"], prefix="season")

    return df



def train_gbdt_timeseries_cv(df: pd.DataFrame,
                              target_col: str = "measurement",
                              feature_cols: list[str] = None,
                              n_splits: int = 3):
    """
    Führt zeitreihen-konforme Cross-Validation mit GBDT durch (kein Data Leakage).

    Args:
        df (pd.DataFrame): Zeitlich sortierter DataFrame mit 'measurement_time', Features und Ziel.
        target_col (str): Zielvariable.
        feature_cols (list[str], optional): Welche Spalten als Features verwendet werden.
        n_splits (int): Anzahl der zeitlichen Folds (z. B. 3).

    Returns:
        model (final trained), list of fold errors (MSEs)
    """
    df = df.dropna(subset=[target_col]).copy()
    df = df.sort_values("measurement_time")

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, "measurement_time"]]

    X = df[feature_cols]
    y = df[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_errors = []
    last_X_test = None
    last_y_test = None

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        last_X_test = X_test
        last_y_test = y_test
        last_test_index = test_index

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        fold_errors.append(mse)
        print(f"Fold {fold+1} MSE: {mse:.3f}")

    # Modell nochmal auf den vollen Datensatz trainieren 
    final_model = GradientBoostingRegressor()
    final_model.fit(X, y)

    return final_model, fold_errors, last_X_test, last_y_test, last_test_index, feature_cols


def detect_anomalies_residuals(model, X_test, y_test, test_index, df, z_thresh: float = 2.5):
    """
    Identifiziert Anomalien anhand der Abweichung zwischen Modellvorhersage und tatsächlichem Wert.

    Args:
        model: trainiertes Regressionsmodell mit predict()-Methode
        X_test (pd.DataFrame): Feature-Daten zur Vorhersage
        y_test (pd.Series): Tatsächliche Zielwerte
        z_thresh (float): Z-Score-Schwelle für Anomalie (z. B. 2.5)

    Returns:
        anomalies_df (pd.DataFrame): Alle Anomalien mit Zeit, Vorhersage, Ist-Wert, Residuum, Z-Score
        residuals_df (pd.DataFrame): Alle Residuen mit Z-Scores
    """
    test_index = test_index
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    residuals_mean = residuals.mean()
    residuals_std = residuals.std()

    z_scores = (residuals - residuals_mean) / residuals_std
    # Anomalie-Wahrscheinlichkeit (sigmoid)
    anomaly_probs = 1 / (1 + np.exp(-z_scores.abs()))

    result_df = pd.DataFrame({
    "measurement_time": df.loc[test_index]["measurement_time"].values, 
    "actual": y_test.values,
    "predicted": y_pred,
    "residual": residuals.values,
    "z_score": z_scores.values,
    "anomaly_score": anomaly_probs.values
})


    anomalies_df = result_df[np.abs(result_df["z_score"]) > z_thresh]
    return anomalies_df, result_df

def loop_forecast_every_5_min():
    # Setup
    box_id = "5ea96b86cc50b1001b78fe27"
    sb = SenseBox(box_id)
    #dbm = DBManagement(box_id)
    #! Parquet-Daten
    # df_train = pd.read_parquet("DF_training_data.parquet")
    # df_temp = df_train[df_train["title"] == "Temperatur"]
    # print(df_temp["measurement_time"].head())
    #! Parquet-Daten Ende
    # Sensor-ID für Temperatur ermitteln
    sensor_infos = sb.get_sensor_info()
    temperature_sensor = next((s for s in sensor_infos if "Temperatur" in s["title"]), None)
    if not temperature_sensor:
        raise ValueError("Kein Temperatursensor gefunden.")
    sensor_id = temperature_sensor["sensor_id"]

    # Einmaliges Modelltraining für festen Zeitraum
    utc = dt.timezone.utc
    datetime_from = dt.datetime(2024, 9, 1, tzinfo=utc)
    datetime_to = dt.datetime(2025, 6, 24, tzinfo=utc)

    df_train = sb.get_sensor_data(sensor_id=sensor_id, datetime_from=datetime_from, datetime_to=datetime_to)
    df_train = resample_measurements(df_train)
    df_train = add_lag_features(df_train).dropna()
    df_train = add_temporal_features(df_train)

    model, _, _, _, _, feature_cols = train_gbdt_timeseries_cv(df_train)  #  feature_cols gespeichert
    print(" Modelltraining abgeschlossen")

    while True:
        print("Starte neuen Vorhersage-Durchlauf...")

        now = dt.datetime.now(dt.timezone.utc)
        past = now - dt.timedelta(minutes=5)
        df_live = sb.get_sensor_data(sensor_id=sensor_id, datetime_from=past, datetime_to=now)

        df_live = resample_measurements(df_live)
        df_live = add_lag_features(df_live).dropna()
        df_live = add_temporal_features(df_live)

        if df_live.empty:
            print(" Keine neuen Daten gefunden.")
        else:
            # Fehlende Features auffüllen
            for col in feature_cols:
                if col not in df_live.columns:
                    df_live[col] = 0
            X_live = df_live[feature_cols]  # Exakte Reihenfolge
            y_live = df_live["measurement"]
            test_index = df_live.index

            anomalies, result_df = detect_anomalies_residuals(
                model, X_live, y_live, test_index, df_live
            )

            print(result_df.tail())
            print(f"Anomalien erkannt: {len(anomalies)}")

            #if not anomalies.empty:
            #    dbm.write_anomalies_to_db(sensor_id, anomalies)

        print("Warten auf nächsten Lauf in 5 Minuten...")
        time.sleep(5 * 60)


# ---------------------------------------------------------------------------------------------------------------#


def main():
    box_id = "5ea96b86cc50b1001b78fe27"
    df = get_temperature_data(box_id)

    df_time_blocks = resample_measurements(df)  
    df_with_lags = add_lag_features(df_time_blocks).dropna()
    df_with_features = add_temporal_features(df_with_lags)

    model, errors, x_test, y_test, test_index = train_gbdt_timeseries_cv(df_with_features)
    anomalies, result_resids = detect_anomalies_residuals(model, x_test, y_test, test_index, df_with_features)

    print(result_resids.head())
    print(anomalies)
    print(f"Anzahl erkannter Anomalien: {len(anomalies)}")

if __name__ == "__main__":
    loop_forecast_every_5_min()
