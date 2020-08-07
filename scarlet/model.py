from abc import ABC, abstractmethod

from .parameter import Parameter


class Model(ABC):
    """Model base class.

    This class stores and provides access to parameters and sub-ordinate models.

    Parameters
    ----------
    parameters: list of `~scarlet.Parameter`
    children: list of `~scarlet.Model`
        Subordinate models.
    """

    def __init__(self, *parameters, children=None):

        if len(parameters) == 0:
            self._parameters = ()
        elif isinstance(parameters, Parameter):
            self._parameters = (parameters,)
        elif isinstance(parameters, (list, tuple)):
            for p in parameters:
                assert isinstance(p, Parameter)
            self._parameters = parameters
        else:
            raise TypeError(
                "parameter must be None, a Parameter, or a list of Parameters"
            )

        if children is None:
            children = ()
        if hasattr(children, "__iter__"):
            for c in children:
                assert isinstance(c, Model)
            self._children = children
        else:
            assert isinstance(children, Model)
            self._children = tuple(children)

        self.check_parameters()

    @property
    def parameters(self):
        """List of parameters, including from the children.
        """
        return self._parameters + tuple(p for c in self.children for p in c.parameters)

    @property
    def children(self):
        return self._children

    def __getitem__(self, i):
        return self._children.__getitem__(i)

    def __iter__(self):
        return self._children.__iter__()

    def __next__(self):
        return self._children.__next__()

    def get_parameter(self, i, *parameters):
        # NOTE: index lookup only works if order is not changed by parameter fixing!
        # check parameters first during optimization
        if i < len(parameters):
            return parameters[i]

        # otherwise use self
        if i < len(self.parameters):
            return self.parameters[i]

        return None

    @abstractmethod
    def get_model(self, *parameters, **kwargs):
        """Get the model realization

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

        Returns
        -------
        model: array
            Realization of the model
        """
        pass

    def get_models_of_children(self, *parameters, **kwargs):
        models = []
        # parameters during optimization
        if len(parameters):
            # count non-fixed own parameters
            i = len(self._parameters)
            for c in self._children:
                j = len(c.parameters)
                models.append(c.get_model(*(parameters[i : i + j]), **kwargs))
                i += j
        else:
            for c in self._children:
                models.append(c.get_model(**kwargs))
        return models

    def check_parameters(self):
        """Check that all parameters have finite elements

        Raises
        ------
        `ArithmeticError` when non-finite elements are present
        """
        for k, p in enumerate(self.parameters):
            if not p.is_finite:
                msg = "Model {}, Parameter '{}' is not finite:\n{}".format(
                    self.__class__.__name__, p.name, p
                )
                raise ArithmeticError(msg)
