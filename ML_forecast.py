from senseboxAPI import SenseBox
import datetime as dt
from datetime import timezone
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
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
    df = df.sort_values("measurement_time")
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

    # Feature: Zeit seit letzter Messung in Minuten
    df = df.sort_values("measurement_time")
    df["minutes_since_last_measurement"] = df["measurement_time"].diff().dt.total_seconds().div(60)
    df["minutes_since_last_measurement"] = df["minutes_since_last_measurement"].fillna(0)

    return df


def add_rolling_features(df, target_col="measurement", window=5):
    df = df.copy()
    df[f"{target_col}_roll_mean"] = df[target_col].rolling(window).mean()
    df[f"{target_col}_roll_std"] = df[target_col].rolling(window).std()
    return df



def train_gbdt_timeseries_cv(df: pd.DataFrame,
                              target_col: str = "measurement",
                              feature_cols: list[str] = None,
                              n_splits: int = 3):
    """
    Führt zeitreihen-konforme Cross-Validation mit GBDT durch.

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
        feature_cols = [col for col in df.columns if col not in [target_col, "measurement_time", "title", "icon", 
                                                                 "sensor_type", "unit", "sensor_id", "box_id"]]

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

        model = XGBRegressor(n_estimators=300,
                            learning_rate=0.05,
                            max_depth=5,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42
                            )


        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        fold_errors.append(mse)
        print(f"Fold {fold+1} MSE: {mse:.3f}")

    # Modell nochmal auf den vollen Datensatz trainieren 
    final_model = XGBRegressor()
    final_model.fit(X, y)

    return final_model, fold_errors, last_X_test, last_y_test, last_test_index, feature_cols, target_col


def detect_anomalies_residuals(model, feature_cols, target_col, test_index, df, z_thresh: float = 3):
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
    original_cols = ["box_id", "sensor_id", "measurement_time", "measurement", "unit", "sensor_type", "title", "icon"]

    X_test = df[feature_cols]
    y_test = df[target_col]
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Robuster Z-Score mit Median und MAD
    z_scores = np.abs(residuals / (np.std(residuals) + 1e-6))

    anomaly_score = 1 - np.exp(-z_scores)


    result_df = df[original_cols].copy()
    result_df.loc[y_test.index, "anomaly_score"] = anomaly_score  # anomaly_scores kommt aus detect_anomalies_residuals
    result_df.loc[y_test.index, "z_score"] = z_scores

    # result_df = pd.DataFrame({
    #     "measurement_time": df.loc[test_index]["measurement_time"].values,
    #     "actual": y_test.values,
    #     "predicted": y_pred,
    #     "residual": residuals.values,
    #     "z_score": z_scores.values,
    #     "anomaly_score": anomaly_score.values
    # })

    anomalies_df = result_df[np.abs(result_df["z_score"]) > z_thresh]
    return anomalies_df, result_df


def main():
    # Feste Box-ID
    box_id = "5ea96b86cc50b1001b78fe27"
    sb = SenseBox(box_id)

    # Trainingsdaten aus Parquet
    df_train = pd.read_parquet("DF_training_data.parquet")
    df_temp = df_train[df_train["title"] == "Temperatur"]

    # Vorbereitung der Daten
    df_temp = resample_measurements(df_temp)
    df_temp = add_rolling_features(df_temp)
    df_temp = add_lag_features(df_temp).dropna()
    df_temp = add_temporal_features(df_temp)
    df_temp["timestamp"] = df_temp["measurement_time"].astype("int64")

    # Modelltraining
    model, errors, x_test, y_test, test_index, feature_cols, target_col = train_gbdt_timeseries_cv(df_temp)
    print("Modelltraining abgeschlossen")

    # Einmalige Anomalie-Erkennung (Testzweck)
    anomalies, result_df = detect_anomalies_residuals(model, feature_cols, target_col, test_index, df_temp)

    # Ausgabe
    print(result_df.tail())
    print(len(result_df))
    print(f"Anzahl erkannter Anomalien: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies)

if __name__ == "__main__":
    main()
