# Simple Variational Autoencoder
This is a simple implementation of Variational Autoencoder.
The encoder and decoders are all fully connected neural networks.

## Use
```python
$python vae_train.py
```

## Example on MNIST Dataset

### Samples
![](https://github.com/wuga214/Variational-Auto-Encoder/blob/master/figs/train/grid/1490.pdf)


### Latent Space Distribution
![](https://github.com/wuga214/Variational-Auto-Encoder/blob/master/figs/train/scatter/1490.pdf)


## Issue
There is an implementation of Gaussian observation VAE.. 
But the training is extremely unstable due to tiny variance learned.
Maybe it is not practical at all.