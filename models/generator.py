import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Normal


class GENERATOR(object):
    def __init__(self, latent_dim, observation_dim, generator,
                 obs_distrib="Bernoulli",
                 obs_std=0.01,
                 ):
        """

        """
        self._latent_dim = latent_dim
        self._observation_dim = observation_dim
        self._generator = generator
        self._obs_distrib = obs_distrib
        self._obs_std = obs_std
        self._p_distribution = self._multivariate_normal()
        self._build_graph()

    def _multivariate_normal(self):
        return Normal([0.] * self._latent_dim, [1.] * self._latent_dim)

    def _build_graph(self):

        with tf.variable_scope('in'):
            # placeholder for the input noise
            self.candid = tf.placeholder(tf.float32, shape=[None, self._latent_dim], name='candidate')

            self.partialx = tf.placeholder(tf.float32, shape=[None, self._observation_dim], name='partialx')
            self.mask = tf.placeholder(tf.float32, shape=[self._observation_dim], name='mask')

            # decode batch
            with tf.variable_scope('generator'):
                self.generated = self._generator(self.candid, self._latent_dim)

            if self._obs_distrib == "Gaussian":
                with tf.variable_scope('gaussian'):
                    log_like = self._gaussian_log_likelihood(self.partialx * self.mask,
                                                             self.generated * self.mask,
                                                             self._obs_std)
            else:
                with tf.variable_scope('bernoulli'):
                    log_like = self._bernoulli_log_likelihood(self.partialx * self.mask,
                                                              self.generated * self.mask)

            self.log_like = log_like - tf.reduce_sum(tf.log(self._p_distribution.prob(self.candid)), axis=1)
            self.probs = tf.exp(-self.log_like)

            with tf.variable_scope('gradient'):
                self.gradient = tf.gradients(tf.reduce_sum(self.log_like), self.candid)

            self._sesh = tf.Session()
            init = tf.global_variables_initializer()
            self._sesh.run(init)

    def load_pretrained(self, path):
        generator_variables = []
        for v in tf.trainable_variables():
            if "generator" in v.name:
                generator_variables.append(v)
        saver = tf.train.Saver(generator_variables)
        saver.restore(self._sesh, path)

    def e2x(self, noise):
        x = self._sesh.run(self.generated,
                           feed_dict={self.candid: noise})
        return x

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean), axis=1) / (2 * tf.square(std))
        return se

    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):
        log_like = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                  + (1. - targets) * tf.log((1. - outputs) + eps), axis=1)
        return log_like

    def get_density(self, candid, partialx, mask):
        density = self._sesh.run(self.probs,
                                 feed_dict={self.candid: candid, self.partialx: partialx, self.mask: mask})
        return density

    def get_log_density(self, candid, partialx, mask):
        log_like = self._sesh.run(self.log_like,
                                  feed_dict={self.candid: candid, self.partialx: partialx, self.mask: mask})
        return log_like

    def get_gradient(self, candid, partialx, mask):
        gradient = self._sesh.run(self.gradient,
                                  feed_dict={self.candid: candid, self.partialx: partialx, self.mask: mask})
        return np.asarray(gradient)[0]