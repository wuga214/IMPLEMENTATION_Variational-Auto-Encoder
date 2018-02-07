import tensorflow as tf
from tensorflow.contrib import layers


def fc_mnist_encoder(x, latent_dim, activation=None):
    e = layers.fully_connected(x, 500, scope='fc-01')
    e = layers.fully_connected(e, 500, scope='fc-02')
    e = layers.fully_connected(e, 200, scope='fc-03')
    e = layers.fully_connected(e, 2 * latent_dim, activation_fn=activation,
                               scope='fc-final')

    return e


def fc_mnist_decoder(z, observation_dim, activation=tf.sigmoid):
    x = layers.fully_connected(z, 200, scope='fc-01')
    x = layers.fully_connected(x, 500, scope='fc-02')
    x = layers.fully_connected(x, 500, scope='fc-03')
    x = layers.fully_connected(x, 2 * observation_dim, activation_fn=activation,
                               scope='fc-final')
    return x