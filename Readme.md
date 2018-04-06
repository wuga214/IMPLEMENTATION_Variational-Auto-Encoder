Variational Autoencoder
===
This is a enhanced implementation of Variational Autoencoder.
Both fully connected and convolutional encoder/decoder are built in this model.
Please star if you like this implementation.

## Use
```python
$python vae_train_amine.py # for training
$python sample.py # for sampling
```

## Update
1. Removed standard derivation learning on Gaussian observation decoder.
2. Set the standard derivation of observation to hyper-parameter.
3. Add deconvolution CNN support for the Anime dataset.
4. Remove Anime dataset itself to avoid legal issues.

## Pre-Trained Models
There are two pretrained models
1. Anime
2. MNIST

The weights of pretrained models are locaded in weights folder

## Samples

### ANIME
![](https://github.com/wuga214/Variational-Auto-Encoder/blob/master/figs/train/grid/anime_samples.png)

### MNIST
![](https://github.com/wuga214/Variational-Auto-Encoder/blob/master/figs/train/grid/mnist_samples.png)


### Latent Space Distribution
![](https://github.com/wuga214/Variational-Auto-Encoder/blob/master/figs/train/scatter/latent.png)
