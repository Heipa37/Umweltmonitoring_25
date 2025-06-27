import datetime as dt
from datetime import timezone, datetime, timedelta
from senseboxAPI import SenseBox
import pandas as pd
import numpy as np
import time
import pytz
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px

from db_management import DBManagement
from senseboxAPI import SenseBox
#import ML_forecast

def get_data_by_id(sensor_id: str):
    dbm = DBManagement(sensor_id)
    dbm.db_setup()
    #dbm.db_reset()
    dbm.write_new_data(datetime_from=dt.datetime.now(tz=timezone.utc)-dt.timedelta(days=60), 
                   datetime_to=dt.datetime.now(tz=timezone.utc)-dt.timedelta(days=30))
    return dbm.read_data()

TITLES = ["Luftdruck", "Luftfeuchtigkeit", "Temperatur", "PM10", "PM2.5", "CO2"]


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Bitte geben Sie eine ID einer senseBox ein"),
    dcc.Input(
        id='input-id',
        type='text',
        placeholder='z. B. 5ea96b86cc50b1001b78fe27',
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
    ],
    value=7,  # Standardwert
    clearable=False,
    style={'width': '200px', 'marginTop': '10px'}
),
    html.Div(id='output-container')
])

@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-id', 'value'),
    Input('days-dropdown', 'value')


)
def handle_id_input(n_clicks, user_input_id, selected_days):
    if n_clicks == 0 or not user_input_id:
        return ""
    data = get_data_by_id(user_input_id)

    # Filtert die Daten nach dem gewählten Zeitraum
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=selected_days)
    data = data[data['measurement_time'] >= cutoff]
    data.sort_values(by='measurement_time', inplace=True)

    graphs = []
    for title in TITLES:
        filtered = data[data['title'] == title]
        if not filtered.empty:
            fig = px.line(
                filtered,
                x="measurement_time",
                y="measurement",
                title=title,
                labels={"measurement_time": "Zeitpunkt", "measurement": f"{title} ({filtered['unit'].iloc[0]})"}
            )
        graphs.append(dcc.Graph(figure=fig))

    if not graphs:
        return html.Div("Keine relevanten Messdaten vorhanden.")

    return graphs

    #return html.Pre(str(data))
    #return f"Eingegebene ID: {user_input_id}"

if __name__ == '__main__':
    app.run(debug=True)
