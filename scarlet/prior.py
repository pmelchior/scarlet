import numpy as np

class Prior:
    """Prior base class

    Priors encode distributions of valid solutions for optimization parrameters.
    """
    use_batch = False

    def __call__(self, *X):
        """Compute the log-likelihood of X under the prior
        """
        pass

    def grad(self, *X):
        """Compute the gradients of the log-likelihood of X under the prior
        """
        pass


try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow_hub as hub

    class PixelCNNPrior(Prior):
        use_batch = True

        def __init__(self, module_name, stamp_size=32):
            """
            Initializes a morphology prior from specified TF Hub module

            Parameters
            ----------
            module_name: `str`
                Path or URL of TF Hub module used to calculate a prior gradient.
            """
            #TODO: read the stamp size from the module
            self.stamp_size = stamp_size
            self.module = hub.Module(module_name)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        def __call__(self, *X):
            """Compute the log-likelihood of X under the prior
            """
            return self.module(*X)

        def grad(self, *X):
            """
            Computes the gradient of the prior

            Parameters
            ----------
            x: `tf.tensor`
                4d Tensor containing the input image (s)
            """
            #TODO: Standardize the keywords for the prior outputs

            batch_size = len(X)
            if batch_size > 0:
                inx = tf.placeholder(shape=[batch_size, self.stamp_size, self.stamp_size, 1],
                                     dtype=tf.float32)
                splits = tf.split(inx, num_or_size_splits=batch_size)
                grad_prior = tf.concat([-self.module(s, as_dict=True)['grads'] for s in splits], axis=0)
                compute_grad_prior = lambda x: self.sess.run(grad_prior, feed_dict={inx: x})
                return compute_grad_prior(*X)

except ImportError:
    pass
