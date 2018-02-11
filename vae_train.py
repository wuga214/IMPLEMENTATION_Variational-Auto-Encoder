from coders.vae_coding import fc_mnist_encoder, fc_mnist_decoder
import tensorflow as tf
import numpy as np
from plots.grid_plots import show_samples, show_latent_scatter
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from models.vae import VAE

"""
This simple implementation is heavily refer on some github code online.
Such as:
https://github.com/kvfrans/variational-autoencoder
https://github.com/hwalsuklee/tensorflow-mnist-VAE
etc

The entire purpose of releasing this code is to help people understand the simple structure of VAE.
"""

def main():
    flags = tf.flags
    flags.DEFINE_integer("latent_dim", 2, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    flags.DEFINE_integer("epochs", 500, "As it said")
    flags.DEFINE_integer("updates_per_epoch", 100, "Really just can set to 1 if you don't like mini-batch.")
    flags.DEFINE_string("data_dir", 'mnist', "Tensorflow demo data download position.")
    FLAGS = flags.FLAGS

    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'encoder': fc_mnist_encoder,
        'decoder': fc_mnist_decoder
    }
    vae = VAE(**kwargs)
    mnist = input_data.read_data_sets(train_dir=FLAGS.data_dir)
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.

        for _ in range(FLAGS.updates_per_epoch):
            x, _ = mnist.train.next_batch(FLAGS.batch_size)
            loss = vae.update(x)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)

    z = np.random.normal(size=[FLAGS.batch_size, FLAGS.latent_dim])
    samples = vae.z2x(z)[0]
    show_samples(samples, 10, 10, [28, 28], name='samples')
    show_latent_scatter(vae, mnist, name='latent')

    vae.save_generator('weights/vae_mnist/generator')


if __name__ == '__main__':
    main()