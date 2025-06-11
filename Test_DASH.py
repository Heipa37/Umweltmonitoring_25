#Nur ein Test 

import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd

backgroundColor = '#023047'
text_color = '#8ECAE6'
text_color_diagram = '#219EBC'  # Farbe für die Diagrammtexte
line_color = '#FFB703'  # Farbe für die Linien in den Diagrammen
line_color2 = '#FB8500'  # Zweite Farbe für die Linien in den Diagrammen

# weitere Farben: 8ECAE6  219EBC  023047  FFB703  FB8500
# 

df = pd.read_csv("umwelt_juni_2025.csv")

# Beispiel-Figure
fig = go.Figure()
fig2 = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Datum'],
    y=df['Temperatur (°C)'],
    mode='lines+markers',
    name='Temperatur (°C)',
    line=dict(color=line_color)
))

fig.add_trace(go.Scatter(
    x=df['Datum'],
    y=df['Luftfeuchtigkeit (%)'],
    mode='lines+markers',
    name='Luftfeuchtigkeit (%)',
    line=dict(color=line_color2)
))

fig2.add_trace(go.Scatter(
    x=df['Datum'],
    y=df['CO2 (ppm)'],
    mode='lines+markers',
    name='CO₂ (ppm)',
    line=dict(color=line_color),
))

# Layout mit Hintergrundfarben
fig.update_layout(
    paper_bgcolor=backgroundColor,  # Hintergrund außenrum (Chart-Hintergrund)
    plot_bgcolor=backgroundColor,
    font= dict(color=text_color),
    title='Testdiagramm mit Hintergrundfarbe',
    yaxis=dict(
        title='Temperatur / Luftfeuchtigkeit',
        #titlefont=dict(color=text_color_diagram),
        tickfont=dict(color=text_color_diagram)
    ),
    xaxis=dict(
        title='Datum',
        tickfont=dict(color=text_color_diagram),
        #tickangle=-45  # Optional: Drehung der X-Achsen-Beschriftungen
    )
)

fig2.update_layout(
    paper_bgcolor=backgroundColor,  # Hintergrund außenrum (Chart-Hintergrund)
    plot_bgcolor=backgroundColor,
    font= dict(color=text_color),
    title='Testdiagramm mit Hintergrundfarbe',
    yaxis=dict(
        title='CO2 Gehalt',
        #titlefont=dict(color=text_color_diagram),
        tickfont=dict(color=text_color_diagram)
    ),
    xaxis=dict(
        title='Datum',
        tickfont=dict(color=text_color_diagram),
        tickangle=-45  # Optional: Drehung der X-Achsen-Beschriftungen
    )
)

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': backgroundColor, 
            'color': text_color, 
            'fontFamily': 'Arial', 
            'minHeight': '100vh',
            'margin': '0',
            'padding': '30px',
            'position': 'absolute',
            'top': '0',
            'left': '0',
            'width': '100%'},
            children=[
                html.H1("Projekt Umweltmonitoring"),
    dcc.Graph(
        id='mein-plot',
        figure=fig
    ),
    dcc.Graph(
        id='mein-plot2',
        figure=fig2
    )
])

app.run(debug=True)