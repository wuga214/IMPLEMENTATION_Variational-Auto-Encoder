import tensorflow as tf
from tensorflow.contrib import layers


def fc_mnist_encoder(x, latent_dim, activation=None):
    e = layers.fully_connected(x, 500, scope='fc-01')
    e = layers.fully_connected(e, 500, scope='fc-02')
    e = layers.fully_connected(e, 200, scope='fc-03')
    output = layers.fully_connected(e, 2 * latent_dim, activation_fn=activation,
                                    scope='fc-final')

    return output


def fc_mnist_decoder(z, observation_dim, activation=tf.sigmoid):
    x = layers.fully_connected(z, 200, scope='fc-01')
    x = layers.fully_connected(x, 500, scope='fc-02')
    x = layers.fully_connected(x, 500, scope='fc-03')
    output = layers.fully_connected(x, observation_dim, activation_fn=activation,
                                    scope='fc-final')
    return output

def conv_anime_encoder(x, latent_dim, activation=None):

    normalizer_params = {'decay': 0.9, 'scale': True, 'center': False}
    e = tf.reshape(x, [-1, 64, 64, 3])
    filter_size = 64
    e = layers.conv2d(e, filter_size, 5,
                      # activation_fn=leaky_rectify,
                      padding='SAME', stride=2, scope='conv-0{0}'.format(0))

    for i in range(2):
        filter_size = filter_size*2
        e = layers.conv2d(e, filter_size, 5,
                          # activation_fn=leaky_rectify,
                          normalizer_fn=tf.contrib.layers.batch_norm,
                          normalizer_params=normalizer_params,
                          padding='SAME', stride=2, scope='conv-0{0}'.format(i + 1))
    e = layers.conv2d(e, 256, 3,
                      # activation_fn=leaky_rectify,
                      normalizer_fn=tf.contrib.layers.batch_norm,
                      normalizer_params=normalizer_params,
                      padding='SAME', stride=1, scope='conv-final')
    e = layers.flatten(e)
    output = layers.fully_connected(e, 2 * latent_dim,
                                    scope='fc-final', activation_fn=activation)

    return output


def conv_anime_decoder(z, observation_dim, activation=tf.tanh):

    filter_size = 256
    normalizer_params = {'decay': 0.9, 'scale': True, 'center': False}

    x = layers.fully_connected(z, 4096,
                               normalizer_fn=tf.contrib.layers.batch_norm,
                               normalizer_params=normalizer_params,
                               scope='fc-01', activation_fn=None)
    x = tf.reshape(x, [-1, 4, 4, filter_size])
    x = layers.conv2d_transpose(x, filter_size, 3,
                                padding='SAME', stride=1, scope='conv-initial')

    for i in range(4):
        filter_size = filter_size / 2
        x = layers.conv2d_transpose(x, filter_size, 5,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params=normalizer_params,
                                    padding='SAME', stride=2, scope='conv-transpose-0{0}'.format(i + 1))
    x = layers.conv2d_transpose(x, 3, 3,
                                padding='SAME', stride=1, scope='conv-transpose-final', activation_fn=None)
    output = activation(layers.flatten(x))

    return output
