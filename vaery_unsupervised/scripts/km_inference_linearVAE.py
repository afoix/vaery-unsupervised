#%% testing
import numpy as np
from pathlib import Path
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    simple_masking, 
    DATASET_NORM_DICT, 
    SpatProteoDatasetZarr,
    SpatProtoZarrDataModule,
)
from vaery_unsupervised.networks.LightningVAE_linear_km import SpatialVAE_Linear
from vaery_unsupervised.networks.km_ryan_linearresnet import (ResNet18Dec, ResNet18Enc)
import yaml
from vaery_unsupervised.km_utils import plot_batch_sample,plot_dataloader_output
import monai.transforms as transforms
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

dataset_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/")


dataset_zarr = SpatProteoDatasetZarr(
    dataset_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=None,
    transform_input=None
)
plot_dataloader_output(dataset_zarr[0])
# %%
lightning_module = SpatProtoZarrDataModule(
    dataset_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=None,
    transform_input=None,
    num_workers=8,
    batch_size=16,
)
lightning_module.setup("predict")

loader = lightning_module.predict_dataloader()
#%%
for batch in loader:
    batch = batch
    break

#%%

#%%
checkpoint_path = "/mnt/efs/aimbl_2025/student_data/S-KM/logs/linear_VAE_fixedfinallinear_attempt3/version_0/checkpoints/epoch=271-step=11696.ckpt"
model = SpatialVAE_Linear.load_from_checkpoint(checkpoint_path=checkpoint_path, strict = True)

#%% Compact model loading with error handling

#%%
import torch

#%%
from vaery_unsupervised.networks.LightningVAE_linear_km import reparameterize

# %%
for batch in loader:
    image_ids = batch["metadata"]['well_id']
    input = batch["input"].to(device = model.device)
    reconstruction, z_mean, z_log_var = model(input[:,model.channels_selection,:,:])
    z = reparameterize(z_mean, z_log_var)
    break


# %%
input[:,[model.channels_selection],:,:].shape
# %%
model.channels_selection
# %%
model.device
# %%
all_image_ids = []
all_input = []
all_reconstruction = []
all_z_mean = []
all_z_log_var = []
all_z = []
#%%
model.eval()
#%%
for batch in loader:
    image_ids = batch["metadata"]['well_id']
    input = batch["input"][:,model.channels_selection,:,:].to(device = model.device)
    reconstruction, z_mean, z_log_var = model(input)  
    z = reparameterize(z_mean, z_log_var)

    all_image_ids.append(image_ids) 
    all_input.append(input.detach().cpu())
    all_reconstruction.append(reconstruction.detach().cpu())
    all_z_mean.append(z_mean.detach().cpu())
    all_z_log_var.append(z_log_var.detach().cpu())
    all_z.append(z.detach().cpu())

all_image_ids = np.concatenate(all_image_ids, axis = 0) 
import torch
all_input = torch.cat(all_input, dim = 0).numpy()
all_reconstruction = torch.cat(all_reconstruction, dim = 0).numpy() 
all_z_mean = torch.cat(all_z_mean, dim = 0).numpy() 
all_z_log_var = torch.cat(all_z_log_var, dim = 0).numpy() 
all_z = torch.cat(all_z, dim = 0).numpy()
    
#%%
all_image_ids.shape
import pandas as pd

#%%
np.arange(1,128)
#%%
df = pd.DataFrame(data = all_z, index = all_image_ids, columns = np.arange(1,129))
#%%
df
# %%
from vaery_unsupervised.plotting_utils import *
from sklearn.decomposition import PCA
# %%
pca = PCA(n_components=50).fit_transform(df, y=None)
pca = pd.DataFrame(pca)
pca.index = all_image_ids
pd.DataFrame(pca)
pca_columnnames = [f'pc{i}' for i in np.arange(1,51)]
pca.columns = pca_columnnames
# %%
import pandas as pd
import io
import base64
import seaborn as sns
from dash import dcc, html, Input, Output, no_update, Dash
import plotly.graph_objects as go

from PIL import Image
import numpy as np
from matplotlib.colors import to_hex

# code taken and modified from: https://dash.plotly.com/dash-core-components/tooltip?_gl=1*9tyg7p*_ga*NDYwMzcxMTAxLjE2Njk3MzgyODM.*_ga_6G7EE0JNSC*MTY3MzI2ODgyOS45LjEuMTY3MzI2OTA0Ni4wLjAuMA..
# under the The MIT License (MIT)

# Copyright (c) 2023 Plotly, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

def get_dash_app_3D_scatter_hover_images(
    dataframe:pd.DataFrame,
    plot_keys:list, 
    hue:str,
    images:np.ndarray,
    additional_info: str = "",
    image_size = 200,
):
    """
    The get_dash_app_3D_scatter_hover_images() function creates a Dash app that displays a 3D 
    scatter plot with hover information for each data point. The hover information consists of 
    an image and a label associated with the data point. The image is retrieved from an array 
    of images passed to the function.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        A Pandas DataFrame containing the data to be plotted.
    plot_keys: list 
        A list of column names in the dataframe that represent the x, y, and z coordinates of 
        the data points.
    hue: str
        A string representing the column name in the dataframe that contains the labels 
        associated with the data points.
    images: np.ndarray
        A numpy array containing the images to be displayed in the hover information.
    additional_info: str
        Column name of information which will be displayed with the hover data
    Returns:
        app: a Dash app object representing the 3D scatter plot with hover information.
    """
    # Create a color map for each categorical value and assigns a color to each data 
    # point based on its category. It then extracts the x, y, and z data from the 
    # input DataFrame, and uses them to create a 3D scatter plot using the 
    # plotly.graph_objects library.
    
    labels = dataframe[hue].to_numpy()
    if labels.dtype.name == 'object':
        unique_labels = sorted(np.unique(labels))
        # Fix: Generate enough colors for all unique labels
        color_map = list(sns.color_palette("tab10", n_colors=len(unique_labels)).as_hex())
        mapping = {value:integer for integer,value in enumerate(unique_labels)}
        colors = [color_map[mapping[label]] for label in labels]
    else:
        color_map = sns.color_palette("rocket",as_cmap=True)
        scaled = np.array((labels - labels.min()) / (labels.max()-labels.min()))
        colors = [to_hex(color_map(val)) for val in scaled]
    
    add_info = ["" for i in range(len(dataframe))]
    if additional_info != "":
        add_info = dataframe[additional_info].to_numpy()

    
    x,y,z = [dataframe[key].to_numpy() for key in plot_keys]

    # Make the plot. 
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            opacity=0.7,
            marker=dict(
                size=5,
                color=colors,
            ))],
    )

    # The plot's hover information is set to "none" and its hover template is set 
    # to None to prevent default hover information from being displayed. The plot's 
    # layout is set to fixed dimensions of 1500x800 pixels.
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(
        autosize=False,
        width=1500,
        height=800,
        scene = dict(
            xaxis_title=plot_keys[0],
            yaxis_title=plot_keys[1],
            zaxis_title=plot_keys[2]
        ),
    )


    # Definition of a JupyterDash application and creates a layout 
    # consisting of a dcc.Graph component for the 3D scatter plot and a dcc.Tooltip 
    # component for the hover information.
    app = Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    # Definition of a callback function that listens for hover events on the 3D scatter 
    # plot and returns the appropriate hover information. When a data point is hovered 
    # over, the callback extracts the point's index and image from the input images array, 
    # converts the image to a base64 encoded string using the np_image_to_base64 helper 
    # function, and returns a html.Div containing the image and the category label of 
    # the hovered data point.
    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = images[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url, style={"width": "100%"},
                ),
                html.P(hue + ": " + str(labels[num]), style={'font-weight': 'bold'}),
                html.P(additional_info + ": " + str(add_info[num]), style={'font-weight': 'bold'})
            ], style={'width': f'{image_size}px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    return app

#TODO Correct Docstring
def get_dash_app_2D_scatter_hover_images(
    dataframe:pd.DataFrame,
    plot_keys:list, 
    hue:str,
    images:np.ndarray,
    additional_info: str = "",
    image_size: int = 200,
    marker_size: int = 20,
):
    """
    The get_dash_app_2D_scatter_hover_images() function creates a Dash app that displays a 2D 
    scatter plot with hover information for each data point. The hover information consists of 
    an image and a label associated with the data point. The image is retrieved from an array 
    of images passed to the function.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        A Pandas DataFrame containing the data to be plotted.
    plot_keys: list 
        A list of column names in the dataframe that represent the x and y coordinates of 
        the data points.
    hue: str
        A string representing the column name in the dataframe that contains the labels 
        associated with the data points.
    images: np.ndarray
        A numpy array containing the images to be displayed in the hover information.
    additional_info: str
        Column name of information which will be displayed with the hover data
    image_size: int
        Size of the preview image displayed when hovering over a datapoint

    Returns:
        app: a Dash app object representing the 3D scatter plot with hover information.
    """
    # Create a color map for each categorical value and assigns a color to each data 
    # point based on its category. It then extracts the x, y, and z data from the 
    # input DataFrame, and uses them to create a 3D scatter plot using the 
    # plotly.graph_objects library.

    labels = dataframe[hue].to_numpy()
    if labels.dtype.name == 'object':
        # Convert all labels to strings to avoid comparison errors
        labels_str = [str(label) for label in labels]
        unique_labels = sorted(set(labels_str))  # Use set() to avoid np.unique() issues
        color_map = list(sns.color_palette("tab10", n_colors=len(unique_labels)).as_hex())
        mapping = {value: integer for integer, value in enumerate(unique_labels)}
        colors = [color_map[mapping[str(label)]] for label in labels]
    else:
        color_map = sns.color_palette("flare",as_cmap=True)
        scaled = np.array((labels - labels.min()) / (labels.max()-labels.min()))
        colors = [to_hex(color_map(val)) for val in scaled]
    
    add_info = ["" for i in range(len(dataframe))]
    if additional_info != "":
        add_info = dataframe[additional_info].to_numpy()
    
    x,y = [dataframe[key].to_numpy() for key in plot_keys]

    # Make the plot. 
    fig = go.Figure(   data=[go.Scatter(
        x=x,
        y=y,
        mode='markers',
        opacity=0.8,
        marker=dict(
            size=marker_size,
            color=colors,
        )
    )])

    # The plot's hover information is set to "none" and its hover template is set 
    # to None to prevent default hover information from being displayed. The plot's 
    # layout is set to fixed dimensions of 1500x800 pixels.
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        scene = dict(
            xaxis_title=plot_keys[0],
            yaxis_title=plot_keys[1],
        )
    )


    # Definition of a JupyterDash application and creates a layout 
    # consisting of a dcc.Graph component for the 3D scatter plot and a dcc.Tooltip 
    # component for the hover information.
    app = Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    # Definition of a callback function that listens for hover events on the 3D scatter 
    # plot and returns the appropriate hover information. When a data point is hovered 
    # over, the callback extracts the point's index and image from the input images array, 
    # converts the image to a base64 encoded string using the np_image_to_base64 helper 
    # function, and returns a html.Div containing the image and the category label of 
    # the hovered data point.
    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = images[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url, style={"width": "100%"},
                ),
                html.P(hue + ": " + str(labels[num]), style={'font-weight': 'bold'}),
                html.P(additional_info + ": " + str(add_info[num]), style={'font-weight': 'bold'})
            ], style={'width': f'{image_size}px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    return app

# Definition of a nested helper function np_image_to_base64 that converts numpy 
# arrays of images into base64 encoded strings for display in HTML.
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


#%%
pca = pca.reset_index().rename(columns = {"index":"label"})
#
#colors = [color_map[mapping[label]] for label in labels]

#%%
app = get_dash_app_2D_scatter_hover_images(pca, hue = "label",images = all_reconstruction, plot_keys = ["pc1", "pc2", "pc3"])
# %%
app.run(
    port=6009,
    mode='external'  # Opens in new browser tab
)
# %%
import seaborn as sns
sns.scatterplot(pca, x = "pc1", y = "pc2", hue = "label")
# %%
