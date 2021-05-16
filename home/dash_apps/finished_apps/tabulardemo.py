import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
from umap import UMAP
import plotly.express as px
import json
from textwrap import dedent as d
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import shap


catboost_titanic_app = DjangoDash('CatboostTitanicClassifier')












