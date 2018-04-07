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

            # decode batch
            with tf.variable_scope('generator'):
                self.generated = self._generator(self.candid, self._latent_dim)

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