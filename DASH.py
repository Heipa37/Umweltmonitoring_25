import datetime as dt
from datetime import timezone
import pandas as pd
import numpy as np
import time
import threading

import dash
from dash import html, dcc, Input, Output, State
from dash import callback_context
import plotly.express as px
import plotly.graph_objects as go

from db_management import DBManagement
import ML_forecast


class DashApp:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.dbm = None
        self.current_anomalies = pd.DataFrame()
        self.model = None
        self.feature_cols = None
        self.user_id = None
        self.selected_days = 7
        self.layout_setup()

    def layout_setup(self):
        self.app.layout = html.Div([
            html.Div([
                html.H2("📡 SenseBox Dashboard", style={'color': '#ffffff'}),
                html.Label("Bitte geben Sie eine ID einer senseBox ein:", style={'color': '#dddddd'}),
                dcc.Input(
                    id='input-id',
                    type='text',
                    placeholder='z. B. 669a877ae3b7f1000824289c',
                    style={
                        'marginRight': '10px',
                        'padding': '8px',
                        'width': '300px',
                        'borderRadius': '6px',
                        'border': '1px solid #555',
                        'backgroundColor': '#2c2c3c',
                        'color': 'white'
                    }
                ),
                html.Button('Absenden', id='submit-button', n_clicks=0,
                            style={
                                'padding': '8px 16px',
                                'backgroundColor': '#2980b9',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'marginTop': '10px'
                            }),
                html.Br(),
                html.Label("Zeitraum wählen:", style={'color': '#dddddd', 'marginTop': '15px'}),
                dcc.Dropdown(
                    id='days-dropdown',
                    options=[
                        {'label': 'Letzte 7 Tage', 'value': 7},
                        {'label': 'Letzte 14 Tage', 'value': 14},
                        {'label': 'Letzte 21 Tage', 'value': 21},
                        {'label': 'Letzte 28 Tage', 'value': 28},
                        {'label': 'Letzte 365 Tage', 'value': 365}
                    ],
                    value=7,
                    clearable=False,
                    style={
                        'width': '250px',
                        'marginTop': '10px',
                        'backgroundColor': '#dddddd',
                        'color': "#2c2c3c"
                    }
                )
            ], style={
                'padding': '20px',
                'borderRadius': '12px',
                'backgroundColor': '#2a2a3d',
                'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.4)',
                'margin': '20px auto',
                'width': '90%',
                'maxWidth': '800px'
            }),
            html.Div(id='output-container', style={'padding': '20px'}),
            dcc.Interval(
                id='interval-update',
                interval=5 * 60 * 1000,
                n_intervals=0
            )
        ], style={'backgroundColor': '#1e1e2f', 'minHeight': '100vh'})

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

        self.model, _, _, _, _, self.feature_cols, _ = ML_forecast.train_gbdt_timeseries_cv(df_train)
        self.feature_cols = list(map(str, self.feature_cols))
        print("Modelltraining abgeschlossen")

    def forecast(self, selected_days=7):
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
            print("Keine neuen Daten gefunden.")
            return pd.DataFrame()

        for col in self.feature_cols:
            if col not in df_live.columns:
                df_live[col] = 0

        X_live = df_live[self.feature_cols]
        test_index = df_live.index

        anomalies, result_df = ML_forecast.detect_anomalies_residuals(
            self.model, self.feature_cols, target_col, test_index, df_live
        )
        self.current_anomalies = anomalies
        return anomalies

    def handle_user_input(self, n_clicks, user_input_id, selected_days):
        if n_clicks == 0 or not user_input_id:
            return ""
        self.user_id = user_input_id
        self.selected_days = selected_days

        self.get_data_by_id(user_input_id, selected_days)
        if selected_days in [365, 0]:
            self.current_anomalies = pd.DataFrame()  # Keine Anomalien/Vorhersage
        else:
            self.model_training()
            self.forecast(selected_days)

        return self.render_dashboard()

    def update_loop(self, n_intervals, selected_days):
        if self.user_id:
            self.selected_days = selected_days
            self.get_data_by_id(self.user_id, self.selected_days)
            if selected_days in [365, 0]:
                self.current_anomalies = pd.DataFrame()
            else:
                self.forecast(self.selected_days)
            return self.render_dashboard()
        return dash.no_update

    def render_dashboard(self):
        data = self.dbm.read_data()
        data = data[data['measurement_time'] >= (dt.datetime.now(tz=timezone.utc) - dt.timedelta(days=self.selected_days))]
        data.sort_values(by='measurement_time', inplace=True)
        data['measurement_time'] = pd.to_datetime(data['measurement_time'])

        df_temp = self.current_anomalies
        graphs = []

        # Wenn KEINE Anomalien/Vorhersage (also alle Daten oder 365 Tage)
        if df_temp.empty:
            for title in data['title'].unique():
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
                fig.update_layout(
                    plot_bgcolor='#1e1e2f',
                    paper_bgcolor='#1e1e2f',
                    font=dict(color="#ffffff"),
                    margin=dict(l=40, r=30, t=60, b=40),
                    hovermode="x unified"
                )
                graphs.append(dcc.Graph(figure=fig))
        else:
            # Temperatur-Plot mit Anomalien
            if "Temperatur" in data['title'].unique():
                zeitgrenze = dt.datetime.now(tz=timezone.utc) - dt.timedelta(days=self.selected_days)
                df_temp = df_temp[df_temp["measurement_time"] >= zeitgrenze]

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
                        y=df_temp["measurement"],
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
                    title="🌡️ Temperatur mit Anomalie-Färbung",
                    xaxis_title="Zeit",
                    yaxis_title="Temperatur (°C)",
                    plot_bgcolor='#1e1e2f',
                    paper_bgcolor='#1e1e2f',
                    font=dict(color="#ffffff"),
                    margin=dict(l=40, r=30, t=60, b=40),
                    hovermode="x unified"
                )
                graphs.append(dcc.Graph(figure=fig))
            # Weitere Messarten (ohne Anomalien)
            for title in data['title'].unique():
                if title == "Temperatur":
                    continue
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
                fig.update_layout(
                    plot_bgcolor='#1e1e2f',
                    paper_bgcolor='#1e1e2f',
                    font=dict(color="#ffffff"),
                    margin=dict(l=40, r=30, t=60, b=40),
                    hovermode="x unified"
                )
                graphs.append(dcc.Graph(figure=fig))

        if not graphs:
            return html.Div("Keine relevanten Messdaten vorhanden.", style={'color': '#ffffff'})
        return graphs


# App starten
if __name__ == '__main__':
    dash_app = DashApp()

    # Callback bei Button-Klick
    @dash_app.app.callback(
        Output('output-container', 'children'),
        [
            Input('submit-button', 'n_clicks'),
            Input('interval-update', 'n_intervals')
        ],
        [
            State('input-id', 'value'),
            State('days-dropdown', 'value')
        ]
    )
    def combined_callback(n_clicks, n_intervals, user_input_id, selected_days):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'submit-button':
            return dash_app.handle_user_input(n_clicks, user_input_id, selected_days)
        elif trigger_id == 'interval-update':
            return dash_app.update_loop(n_intervals, selected_days)
        return dash.no_update

    dash_app.app.run(debug=True)
