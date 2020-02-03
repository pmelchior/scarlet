from abc import ABC, abstractmethod
import numpy as np


class Prior(ABC):
    """Prior base class

    A prior encodes the distribution of valid solutions for optimization parrameters.
    """

    @abstractmethod
    def __call__(self, *X):
        """Compute the log-likelihood of `X` under the prior
        """
        pass

    @abstractmethod
    def grad(self, *X):
        """Compute the gradients of the log-likelihood of `X` under the prior
        """
        pass

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow_hub as hub

    class PixelCNNPrior(Prior):

        def __init__(self, module_name, stamp_size=32):
            """
            Initializes a morphology prior from specified TF Hub module
            Parameters
            ----------
            module_name: `str`
                Path or URL of TF Hub module used to calculate a prior gradient.
            """
            self.stamp_size = stamp_size
            self.module = hub.Module(module_name)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        def __call__(self, *X):
            """Compute the log-likelihood of X under the prior
            """
            single_shape = (1, self.stamp_size, self.stamp_size, 1)
            return tuple([ self.module(x.reshape(single_shape), as_dict=True)['log_prob'].eval(session=self.sess) for x in X])

        def grad(self, *X):
            """
            Computes the gradient of the prior
            Parameters
            ----------
            x: `tf.tensor`
                4d Tensor containing the input image (s)
            """
            print ("prior", len(tuple(*X)), tuple(*X)[0].shape)

            single_shape = (1, self.stamp_size, self.stamp_size, 1)
            x_shape = (self.stamp_size, self.stamp_size)
            return tuple([self.module(x.reshape(single_shape), as_dict=True)['grads'].eval(session=self.sess).reshape(x_shape) for x in tuple(*X)])

            # TODO: GPU deployment optimimzation: this below seems slower ...?
            # grad_prior = tf.concat([-self.module(x.reshape(single_shape), as_dict=True)['grads'] for x in X], axis=0)
            # return self.sess.run(grad_prior)

except ImportError:
    pass
