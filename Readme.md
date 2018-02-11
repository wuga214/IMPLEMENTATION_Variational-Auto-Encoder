# Simple Variational Autoencoder
This is a simple implementation of Variational Autoencoder.
The encoder and decoders are all fully connected neural networks.

## Use
```python
$python vae_train.py
```

## Update
1. Removed standard derivation learning on Gaussian observation decoder.
2. Set the standard derivation of observation to hyper-parameter.
3. Add deconvolution CNN support for the Anime dataset.
4. Remove Anime dataset itself to avoid legal issues.

## Example on MNIST Dataset

### Samples
![](https://github.com/wuga214/Variational-Auto-Encoder/blob/master/figs/train/grid/samples.png)


### Latent Space Distribution
![](https://github.com/wuga214/Variational-Auto-Encoder/blob/master/figs/train/scatter/latent.png)
