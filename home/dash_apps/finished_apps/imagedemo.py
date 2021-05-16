from djangoandplotly.settings import BASE_DIR
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
from umap import UMAP
import plotly.express as px
import json, os
from textwrap import dedent as d
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
import io, base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# from jupyter_dash import JupyterDash

# app info
# app = JupyterDash(__name__)
mnist_fashion_app = DjangoDash("MnistFashionDeepExplainer")

styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

# data and basic figure
# df = px.data.iris()
def find(proj_3d, x, y, z):
    a = list(proj_3d)
    print(type(a))
    n = [x, y, z]
    for i in range(len(a)):
        if x == a[i][0] and y == a[i][1] and z == a[i][2]:
            return i


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


(
    mnist_fashion_trainX,
    mnist_fashion_trainY,
    mnist_fashion_testX,
    mnist_fashion_testY,
) = load_dataset()

mnist_fashion_model = load_model(
    os.path.join(BASE_DIR, "utils/model_files/mnist_fashion.h5")
)
img = mnist_fashion_trainX[0]
fig = plt.imshow(img, cmap="gray")
my_stringIObytes = io.BytesIO()
fig.figure.savefig(my_stringIObytes, format="png")
my_stringIObytes.seek(0)
my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
data_base64 = my_base64_jpgData.decode()
src = f'<img src="data:image/png;base64, {data_base64}"/>'
features = mnist_fashion_trainX
nsamples, nx, ny, _ = features.shape
d2_train_dataset = features.reshape((nsamples, nx * ny))
umap_3d = UMAP(n_components=3, init="random", random_state=0)
proj_3d = umap_3d.fit_transform(d2_train_dataset)

t = np.asarray([np.argmax(i) for i in mnist_fashion_trainY])
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
output = np.asarray([classes[i] for i in t])
fig_3d = px.scatter_3d(
    proj_3d,
    x=0,
    y=1,
    z=2,
    color=output,
)
fig_3d.update_traces(marker_size=5)

mnist_fashion_app.layout = html.Div(
    [
        dcc.Graph(
            id="training",
            figure=fig_3d,
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    [
                        html.Iframe(
                            id="hover-data",
                            srcDoc=src,
                            style={
                                "frameborder": "0",
                                "overflow": "hidden",
                                "overflow-x": "hidden",
                                "overflow-y": "hidden",
                                "height": "100%",
                                "width": "100%",
                            },
                        ),
                        html.Div(
                            [
                                html.Iframe(
                                    id="shap",
                                    className="embed-responsive-item",
                                    srcDoc="SHAP Force Plots",
                                    style={
                                        "frameborder": "0",
                                        "overflow": "hidden",
                                        "overflow-x": "hidden",
                                        "overflow-y": "hidden",
                                        "height": "100%",
                                        "width": "100%",
                                    },
                                ),
                            ],
                            className="three columns",
                            style={"width": "100%"},
                        ),
                    ],
                    className="three columns",
                ),
            ],
        ),
    ]
)


@mnist_fashion_app.callback(
    Output("hover-data", "srcDoc"),
    Output("shap", "srcDoc"),
    [Input("training", "clickData")],
)
def display_hover_data(clickData):
    x = clickData["points"][0]["x"]
    y = clickData["points"][0]["y"]
    z = clickData["points"][0]["z"]
    index = find(proj_3d, x, y, z)
    img = mnist_fashion_trainX[index]
    fig = plt.imshow(img)
    my_stringIObytes = io.BytesIO()
    fig.figure.savefig(my_stringIObytes, format="png")
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    data_base64 = my_base64_jpgData.decode()
    e = shap.DeepExplainer(mnist_fashion_model, mnist_fashion_trainX[1:1000])
    shap_values = e.shap_values(mnist_fashion_testX[1:8])
    # plot the feature attributions
    plot= shap.image_plot(shap_values, mnist_fashion_testX[1:8], labels = mnist_fashion_testY[1:8]).save
    return (
        f'<img src="data:image/png;base64, {data_base64}" width=300px height=300px/>',
    )
