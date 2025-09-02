import matplotlib.pyplot as plt
import numpy as np



def plot_dataloader_output(data_dict, figsize=(20, 10)):
    """Plot images from data loader output, showing last 3 channels as RGB."""
    
    image_keys = [k for k in ["raw", "input", "target"] if k in data_dict]
    fig, axes = plt.subplots(1, len(image_keys), figsize=figsize)
    if len(image_keys) == 1:
        axes = [axes]
    
    for ax, key in zip(axes, image_keys):
        # Convert tensor to numpy and take last 3 channels: (4,y,x) -> (y,x,3)
        img = data_dict[key].detach().cpu().numpy()[-3:].transpose(1, 2, 0)
        
        # Normalize to [0,1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        ax.imshow(img)
        ax.set_title(key.capitalize())
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_batch_sample(batch_dict, figsize=(12, 4), image_keys = ["raw","input","target"]):
    """Plot single sample from batched data loader output."""
    for idx in range(len(batch_dict["raw"])):
        sample = {k: v[idx] for k, v in batch_dict.items() if k in image_keys}
        plot_dataloader_output(sample, figsize)