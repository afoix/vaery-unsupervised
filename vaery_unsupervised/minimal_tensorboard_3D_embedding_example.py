# %%
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import datasets
digits = datasets.load_digits()
digits.keys()
#%%
# Create a SummaryWriter log directory
writer = SummaryWriter("~/embedding_demo")
images = digits['images']
# Example embeddings: 100 samples, 3-dimensional
embeddings = images.reshape(images.shape[0],images.shape[1]*images.shape[2])

# Example metadata (labels for each embedding point)
labels = digits['target']
images = np.repeat(images[:,np.newaxis,:,:],3,1)
print(images.shape)
# Save embeddings for TensorBoard projector
writer.add_embedding(
    mat=embeddings,
    metadata=labels,
    tag="example",
    label_img=torch.Tensor(images),
)

writer.close()