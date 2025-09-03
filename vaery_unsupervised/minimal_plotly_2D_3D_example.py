#%%
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=1000, n_features=3, centers=4, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None, return_centers=False)
data
# %%
df = pd.DataFrame(data=data[0],columns=["0","1","2"])
df["Label"] = data[1]

images = (np.random.uniform(size=(1000,28,28,3))*255).astype('uint8')
images
# %%
from vaery_unsupervised.plotting_utils import *

app = get_dash_app_2D_scatter_hover_images(df,plot_keys=["0","1"],hue="Label", images=images)

# %%

app.run(
    port=6006
)
# %%
app = get_dash_app_3D_scatter_hover_images(df,plot_keys=["0","1","2"],hue="Label", images=images)

# %%

app.run(
    port=6006
)
# %%
