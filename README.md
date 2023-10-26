# VAE Lib - Variational Autoencoder Library

Welcome to VAE Lib, a Python library for implementing Variational Autoencoders (VAEs) and related models. This library is designed to provide a framework for building and experimenting with various VAE models and components.

## Project Structure

The project is organized into the following directory structure:

```
vae_lib/
│
├── layers/
│   ├── distributions/
│   ├── regressors/
│   ├── __init__.py
│   └── base.py
│
├── models/
│   ├── __init__.py
│   ├── base_models.py
│   ├── sparse_vae.py
│   ├── sparse_vsc.py
│   ├── stochastic_vae.py
│   ├── types.py
│   ├── vamp_prior_vae.py
│   ├── variational_auto_encoder.py
│   └── variational_sparse_coding.py
│
├── README.md
├── LICENSE
└── requirements.txt
```

## Directory Descriptions

- **layers/**: This directory contains submodules related to the layers used in VAEs, including distributions and regressors. These modules provide essential components for building VAE models.

- **models/**: This directory houses the VAE models and related components, including base models, different VAE variants, and related types.

## Installation

To install VAE Lib and its dependencies, you can use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```


