import datetime as dt
from datetime import timezone
import pandas as pd
import numpy as np
import time
import threading

import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from db_management import DBManagement
import ML_forecast


class DashApp:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.dbm = None  # wird bei neuer ID gesetzt
        self.layout_setup()

    def layout_setup(self):
        self.app.layout = html.Div([
            html.H2("Bitte geben Sie eine ID einer senseBox ein"),
            dcc.Input(
                id='input-id',
                type='text',
                placeholder='z. B. 669a877ae3b7f1000824289c',
                style={'marginRight': '10px'},
                size='40'
            ),
            html.Button('Absenden', id='submit-button', n_clicks=0),
            dcc.Dropdown(
                id='days-dropdown',
                options=[
                    {'label': 'Letzte 7 Tage', 'value': 7},
                    {'label': 'Letzte 14 Tage', 'value': 14},
                    {'label': 'Letzte 21 Tage', 'value': 21},
                    {'label': 'Letzte 28 Tage', 'value': 28},
                    {'label': 'Letzte 365 Tage', 'value': 365},
                    {'label': 'Alle Daten', 'value': 0},
                ],
                value=7,
                clearable=False,
                style={'width': '200px', 'marginTop': '10px'}
            ),
            html.Div(id='output-container')
        ])

    def get_data_by_id(self, sensor_id: str, pdays: int = 7) -> pd.DataFrame:
        self.dbm = DBManagement(sensor_id)
        self.dbm.db_setup()
        if pdays == 0:
            self.dbm.write_new_data()
        else:
            self.dbm.write_new_data(
                datetime_from=dt.datetime.now(tz=timezone.utc) - dt.timedelta(days=365),
                datetime_to=dt.datetime.now(tz=timezone.utc)
            )
        df = self.dbm.read_data()
        return df[df['measurement_time'] >= (dt.datetime.now(tz=timezone.utc) - dt.timedelta(days=pdays))]

    def model_training(self):
        df_train = self.dbm.read_data()
        df_train = df_train[df_train["title"] == "Temperatur"]
        df_train["measurement_time"] = pd.to_datetime(df_train["measurement_time"])
        df_train = ML_forecast.resample_measurements(df_train)
        df_train = ML_forecast.add_lag_features(df_train).dropna()
        df_train = ML_forecast.add_temporal_features(df_train)

        model, _, _, _, _, feature_cols, _ = ML_forecast.train_gbdt_timeseries_cv(df_train)
        print("Modelltraining abgeschlossen")
        feature_cols = list(map(str, feature_cols))
        return model, feature_cols

    def forecast(self, model, feature_cols, selected_days=7):
        print("Starte neuen Vorhersage-Durchlauf...")

        target_col = "measurement"
        
        self.dbm.write_new_data(
            datetime_from=dt.datetime.now(tz=timezone.utc) - dt.timedelta(days=selected_days),
            datetime_to=dt.datetime.now(tz=timezone.utc)
        )
        df_live = self.dbm.read_data()
        df_live = df_live[df_live['title'] == "Temperatur"]
        df_live["measurement_time"] = pd.to_datetime(df_live["measurement_time"])

        df_live = ML_forecast.resample_measurements(df_live)
        df_live = ML_forecast.add_lag_features(df_live).dropna()
        df_live = ML_forecast.add_temporal_features(df_live)

        if df_live.empty:
            print(" Keine neuen Daten gefunden.")
            return pd.DataFrame()  # oder return None, je nach Erwartung
        else:
            # Fehlende Features auffüllen
            for col in feature_cols:
                if col not in df_live.columns:
                    df_live[col] = 0
                    print(col)

            df = df_live.copy()
            X_live = df_live[feature_cols]  # Exakte Reihenfolge
            y_live = df_live["measurement"]
            test_index = df_live.index


        anomalies, result_df = ML_forecast.detect_anomalies_residuals(
            model, feature_cols, target_col, test_index, df
        )
        return anomalies

    def loop_forecast_every_5_min(self):
        while True:
            try:
                if self.dbm:
                    model, feature_cols = self.model_training()
                    current_anomalies_df = self.forecast(model, feature_cols)
                    print("Anomalien erkannt:", current_anomalies_df.shape[0])
            except Exception as e:
                print("Fehler in Loop:", e)
            time.sleep(5 * 60)


def handle_id_input(n_clicks, user_input_id, selected_days):
    if n_clicks == 0 or not user_input_id:
        return ""

    dash_app.get_data_by_id(user_input_id, selected_days)
    data = dash_app.dbm.read_data()
    data.sort_values(by='measurement_time', inplace=True)
    data['measurement_time'] = pd.to_datetime(data['measurement_time'])

    model, feature_cols = dash_app.model_training()
    df_temp = dash_app.forecast(model, feature_cols = feature_cols, selected_days = selected_days)

    graphs = []

    for title in data['title'].unique():
        if title == "Temperatur" and not df_temp.empty:
            z_abs = np.abs(df_temp["z_score"])
            z_scaled = (z_abs - z_abs.min()) / (z_abs.max() - z_abs.min() + 1e-6)

            def z_to_color(z_norm):
                r = int(255 * z_norm)
                g = int(255 * (1 - z_norm))
                return f'rgb({r},{g},0)'

            colors = [z_to_color(z) for z in z_scaled]

            fig = go.Figure(
                data=go.Scatter(
                    x=df_temp["measurement_time"],
                    y=df_temp["measurement"],   # <-- hier geändert
                    mode='lines+markers',
                    line=dict(color='lightgray'),
                    marker=dict(
                        color=colors,
                        size=8
                    ),
                    name="Temperatur"
                )
            )

            fig.update_layout(
                title="Temperatur mit Anomalie-Färbung",
                xaxis_title="Zeit",
                yaxis_title="Temperatur (°C)",
            )
        else:
            filtered = data[data['title'] == title]
            if filtered.empty:
                continue
            fig = px.line(
                filtered,
                x="measurement_time",
                y="measurement",
                title=title,
                labels={
                    "measurement_time": "Zeitpunkt",
                    "measurement": f"{title} ({filtered['unit'].iloc[0]})"
                }
            )

        graphs.append(dcc.Graph(figure=fig))

    if not graphs:
        return html.Div("Keine relevanten Messdaten vorhanden.")
    return graphs


# App starten
if __name__ == '__main__':
    dash_app = DashApp()

    # Callback-Registrierung
    dash_app.app.callback(
        Output('output-container', 'children'),
        Input('submit-button', 'n_clicks'),
        State('input-id', 'value'),
        Input('days-dropdown', 'value')
    )(handle_id_input)

    # Starte Loop in Thread
    threading.Thread(target=dash_app.loop_forecast_every_5_min, daemon=True).start()

    # Starte Webserver
    dash_app.app.run(debug=True)
