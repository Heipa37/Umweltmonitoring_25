import datetime as dt
from datetime import timezone, datetime, timedelta
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px

from db_management import DBManagement

class SenseBoxDashboard:
    TITLES = ["Luftdruck", "Luftfeuchtigkeit", "Temperatur", "PM10", "PM2.5", "CO2"]

    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def get_data_by_id(self, sensor_id: str):
        dbm = DBManagement(sensor_id)
        dbm.db_setup()
        dbm.write_new_data(datetime_from=dt.datetime.now(tz=timezone.utc)-dt.timedelta(days=60), 
                           datetime_to=dt.datetime.now(tz=timezone.utc)-dt.timedelta(days=30))
        return dbm.read_data()

    def setup_layout(self):
        self.app.layout = html.Div(style={
            'fontFamily': 'Arial, sans-serif',
            'maxWidth': '900px',
            'margin': 'auto',
            'padding': '20px',
            'minHeight': '100vh'
        }, children=[
            html.H1("SenseBox Messdaten Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
            html.Div([
                dcc.Input(
                    id='input-id',
                    type='text',
                    placeholder='z. B. 5ea96b86cc50b1001b78fe27',
                    style={
                        'marginRight': '10px',
                        'padding': '8px',
                        'width': '300px',
                        'fontSize': '16px'
                    }
                ),
                html.Button('Absenden', id='submit-button', n_clicks=0, style={
                    'padding': '9px 15px',
                    'fontSize': '16px',
                    'marginRight': '20px',
                    'cursor': 'pointer'
                }),
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
                    style={
                        'width': '180px',
                        'display': 'inline-block',
                        'verticalAlign': 'middle',
                        'fontSize': '16px'
                    }
                ),
            ], style={'textAlign': 'center', 'marginBottom': '40px'}),

            html.Div(id='output-container')
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('output-container', 'children'),
            Input('submit-button', 'n_clicks'),
            State('input-id', 'value'),
            Input('days-dropdown', 'value')
        )
        def handle_id_input(n_clicks, user_input_id, selected_days):
            if n_clicks == 0 or not user_input_id:
                return ""

            data = self.get_data_by_id(user_input_id)

            cutoff = datetime.now(timezone.utc) - timedelta(days=selected_days)
            data = data[data['measurement_time'] >= cutoff]
            data.sort_values(by='measurement_time', inplace=True)

            graphs = []

            for title in self.TITLES:
                filtered = data[data['title'] == title]
                if not filtered.empty:
                    fig = px.line(
                        filtered,
                        x="measurement_time",
                        y="measurement",
                        title=f"{title} Messwerte",
                        labels={"measurement_time": "Zeitpunkt", "measurement": f"{title} ({filtered['unit'].iloc[0]})"},
                    )
                    graphs.append(dcc.Graph(figure=fig, style={'marginBottom': '40px'}))

            if not graphs:
                return html.Div("Keine relevanten Messdaten vorhanden.", style={
                    'textAlign': 'center',
                    'fontSize': '18px',
                    'marginTop': '20px'
                })

            return graphs

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    dashboard = SenseBoxDashboard()
    dashboard.run()
