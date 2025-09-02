# VAEry Unsupervised

Representation learning for bioimaging using Variational Autoencoders (VAE) and contrastive learning methods for the AI@MBL course. 

## Installation

<img src="vaery_unsupervised/docs/figures/steve.png" alt="VAEry Unsupervised Overview" width="300"/>

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/afoix/vaery-unsupervised.git
cd vaery-unsupervised

# Create a conda environment with python >3.11
conda create -n vaery python=3.11

# Install in development mode
pip install -e .

# Optional: Install with visualization dependencies
pip install -e .[visual]
```

## Quick Start

```bash
# Run the CLI tool
vaery-unsupervised --help

# Example usage (placeholder - update with actual commands)
vaery-unsupervised train --config config.yaml
```