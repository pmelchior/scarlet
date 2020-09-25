from abc import ABC, abstractmethod

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
