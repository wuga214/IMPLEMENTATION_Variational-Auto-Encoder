from coders.vae_coding import conv_anime_decoder, conv_anime_encoder
import tensorflow as tf
import numpy as np
from plots.grid_plots import show_samples, show_latent_scatter
from providers.anime import Anime
from tqdm import tqdm
from models.generator import GENERATOR

def main():
    flags = tf.flags
    flags.DEFINE_integer("latent_dim", 64, "Dimension of latent space.")
    flags.DEFINE_integer("obs_dim", 12288, "Dimension of observation space.")
    flags.DEFINE_integer("batch_size", 60, "Batch size.")
    flags.DEFINE_integer("epochs", 500, "As it said")
    flags.DEFINE_integer("updates_per_epoch", 100, "Really just can set to 1 if you don't like mini-batch.")
    FLAGS = flags.FLAGS

    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'observation_dim': FLAGS.obs_dim,
        'generator': conv_anime_decoder,
        'obs_distrib': 'Gaussian'
    }
    g = GENERATOR(**kwargs)
    g.load_pretrained("weights/vae_anime/generator")

    z = np.random.normal(size=[FLAGS.batch_size, FLAGS.latent_dim])
    samples = g.e2x(z)
    print samples.shape
    show_samples(samples, 4, 15, [64, 64, 3], name='small_samples', shift=True)

if __name__ == '__main__':
    main()