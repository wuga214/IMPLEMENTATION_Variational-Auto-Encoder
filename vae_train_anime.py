from coders.vae_coding import conv_anime_decoder, conv_anime_encoder
import tensorflow as tf
import numpy as np
from plots.grid_plots import show_samples, show_latent_scatter
from providers.anime import Anime
from tqdm import tqdm
from models.vae import VAE


def main():
    flags = tf.flags
    flags.DEFINE_integer("latent_dim", 64, "Dimension of latent space.")
    flags.DEFINE_integer("obs_dim", 12288, "Dimension of observation space.")
    flags.DEFINE_integer("batch_size", 64, "Batch size.")
    flags.DEFINE_integer("epochs", 500, "As it said")
    flags.DEFINE_integer("updates_per_epoch", 100, "Really just can set to 1 if you don't like mini-batch.")
    FLAGS = flags.FLAGS

    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'observation_dim': FLAGS.obs_dim,
        'encoder': conv_anime_encoder,
        'decoder': conv_anime_decoder,
        'observation_distribution': 'Gaussian'
    }
    vae = VAE(**kwargs)
    provider = Anime()
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.

        for _ in range(FLAGS.updates_per_epoch):
            x = provider.next_batch(FLAGS.batch_size)
            loss = vae.update(x)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)

    z = np.random.normal(size=[FLAGS.batch_size, FLAGS.latent_dim])
    samples = vae.z2x(z)[0]
    show_samples(samples, 8, 8, [64, 64, 3], name='samples')

    vae.save_generator('weights/vae_anime/generator')


if __name__ == '__main__':
    main()