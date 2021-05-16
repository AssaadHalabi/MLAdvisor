from helpers.plots import text_plot
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
from umap import UMAP
import plotly.express as px
import transformers
import shap


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DjangoDash('TextDemo')

# app.layout = html.Div([
#     # html.H1('Square Root Slider Graph'),
#     dcc.Input(id="text_for_sentiment_analysis", placeholder='Enter a value...', type='text', value=''),
#     # dcc.Graph(id='slider-graph', figure = fig_3d, style={"backgroundColor": "#1a2d46", 'color': '#ffffff'}),
#     html.Br(),
#     html.Div(id="text_shap"),
#     # dcc.Slider(
#     #     id='slider-updatemode',
#     #     marks={i: '{}'.format(i) for i in range(20)},
#     #     max=20,
#     #     value=2,
#     #     step=1, 
#     #     updatemode='drag',
#     # ),
# ])




# @app.callback(
#                Output('text_shap', 'children'),
#               [Input('text_for_sentiment_analysis', 'value')])
# def display_shap_picture(value):
    
    

    
    


import json
from textwrap import dedent as d
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
# from jupyter_dash import JupyterDash

# app info
# app = JupyterDash(__name__)

# styles = {
#     'pre': {
#         'border': 'thin lightgrey solid',
#         'overflowX': 'scroll'
#     }
# }

# # data and basic figure
# x = np.arange(20)+10

# fig = go.Figure(data=go.Scatter(x=x, y=x**2, mode = 'lines+markers'))
# fig.add_traces(go.Scatter(x=x, y=x**2.2, mode = 'lines+markers'))
model = transformers.pipeline('sentiment-analysis', return_all_scores=True)


app.layout = html.Div(className='row w-100', children=[
    dcc.Input(className="col-md-10",id="basic-interactions", value='This movie is great, if you have no taste', style={'width':'300px'}),
    html.Br(),
    html.Button('Submit', id='submit-val', n_clicks=0, style={'width':'200px'}),
    html.Br(),
    html.Br(),

    html.Div(className='col-md-12', children=[
        html.Div([
            html.Iframe(id='hover-data',className="embed-responsive-item", srcDoc="", style={
                'frameborder':"0",
                'overflow': 'hidden',
                'overflow-x': 'hidden',
                'overflow-y': 'hidden',
                'height':"100%",
                'width':"100%"}),
        ], className='three columns', style={'width':'100%'}),
    ], style={'width':'100%'})
    
])


@app.callback(
    Output('hover-data', 'srcDoc'),
    [Input('submit-val', 'n_clicks')],
    [State('basic-interactions', 'value')])
def display_hover_data(n_clicks, value):

    # explain the model on two sample inputs
    explainer = shap.Explainer(model) 
    shap_values = explainer([value])

    # visualize the first prediction's explanation for the POSITIVE output class
    return text_plot(shap_values[0, :, "POSITIVE"])._repr_html_()


# app.clientside_callback(
#     """
#     function(figure, scale) {
#         if(figure === undefined) {
#             return {'data': [], 'layout': {}};
#         }
#         const fig = Object.assign({}, figure, {
#             'layout': {
#                 ...figure.layout,
#                 'yaxis': {
#                     ...figure.layout.yaxis, type: scale
#                 }
#              }
#         });
#         return fig;
#     }
#     """,
#     Output('clientside-graph-px', 'figure'),
#     Input('clientside-figure-store-px', 'data'),
#     Input('clientside-graph-scale-px', 'value')
# )