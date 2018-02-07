import re
import numpy as np
import tensorflow as tf


class VAE(object):

    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 observation_dim=784, learning_rate=1e-4, objective='cross-entropy'):

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._observation_dim = observation_dim
        self._learning_rate = learning_rate
        self._objective = objective
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('vae'):
            self.x = tf.placeholder(tf.float32, shape=[None, self._observation_dim])

            with tf.variable_scope('encoder'):
                encoded = self._encode(self.x, self._latent_dim)

            with tf.variable_scope('latent'):
                self.mean = encoded[:, :self._latent_dim]
                logvar = encoded[:, self._latent_dim:]
                stddev = tf.sqrt(tf.exp(logvar))
                epsilon = tf.random_normal([self._batch_size, self._latent_dim])
                self.z = self.mean + stddev * epsilon

            with tf.variable_scope('decoder'):
                decoded = self._decode(self.z, self._observation_dim)

                self.obs_mean = decoded[:, :self._observation_dim]
                obs_logvar = decoded[:, self._observation_dim:]
                obs_var = tf.exp(obs_logvar)
                obs_epsilon = tf.random_normal([self._batch_size, self._observation_dim])
                self.sample = self.obs_mean + tf.sqrt(obs_var) * obs_epsilon

            with tf.variable_scope('loss'):
                with tf.variable_scope('kl-divergence'):
                    kl = self._kl_diagnormal_stdnormal(self.mean, logvar)

                if self._objective == 'squared-error':
                    with tf.variable_scope('squared-error'):
                        obj = self._gaussian_log_likelihood(self.x, self.obs_mean, obs_var)
                else:
                    with tf.variable_scope('cross-entropy'):
                        obj = self._bernoulli_log_likelihood(self.x, self.obs_mean)

                self._loss = (kl + obj) / self._batch_size

            with tf.variable_scope('optimizer'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            self._sesh = tf.Session()
            init = tf.global_variables_initializer()
            self._sesh.run(init)

    @staticmethod
    def _kl_diagnormal_stdnormal(mu, log_var):

        var = tf.exp(log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, var, eps=1e-4):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean) / (var+eps))
        return se

    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):

        log_like = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                  + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like

    def update(self, x):
        _, loss = self._sesh.run([self._train, self._loss], feed_dict={self.x: x})
        return loss

    def x2z(self, x):

        mean = self._sesh.run([self.mean], feed_dict={self.x: x})

        return np.asarray(mean).reshape(-1, self._latent_dim)

    def z2x(self, z):

        x = self._sesh.run([self.obs_mean], feed_dict={self.z: z})
        return x

    def sample(self, z):

        x = self._sesh.run([self.sample], feed_dict={self.z: z})
        return x

    def save_generator(self, path, prefix="in/generator"):
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "decoder" in v.name:
                name = prefix+re.sub("vae/decoder", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)