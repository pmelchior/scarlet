import tensorflow as tf
import tensorflow_hub as hub

class MorphologyPrior():

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

    def grad(self, x):
        """
        Computes the gradient of the prior

        Parameters
        ----------
        x: `tf.tensor`
            4d Tensor containing the input image (s)
        """
        #TODO: Standardize the keywords for the prior outputs
        return (1e-4)*self.module(x, as_dict=True)['grads']
