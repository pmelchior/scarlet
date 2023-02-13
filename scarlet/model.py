from abc import ABC, abstractmethod

from .parameter import Parameter
from autograd.numpy.numpy_boxes import ArrayBox


class UpdateException(Exception):
    pass


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

        #self.check_parameters()

    @property
    def parameters(self):
        """List of parameters, including from the children
        """
        return self._parameters + tuple(p for c in self.children for p in c.parameters)

    @property
    def children(self):
        """List of child models
        """
        return self._children

    def __getitem__(self, i):
        return self._children.__getitem__(i)

    def __iter__(self):
        return self._children.__iter__()

    def __next__(self):
        return self._children.__next__()

    def get_parameter(self, i, *parameters):
        """Access parameters by list index or by name

        Parameters
        ----------
        i: int, slice, str
            Index, slice or name attribute of the requested parameter
        parameters: tuple
            Parameters used during optimization. If not set, uses `self`

        Returns
        -------
        Matching item or tuple of matching items
        """

        # NOTE: index lookup only works if order is not changed by parameter fixing!
        # during optimization: parameters are passed by autograd
        if parameters:
            parameters_ = parameters
        else:
            parameters_ = self.parameters

        if isinstance(i, (int, slice)):
            return parameters_[i]
        elif isinstance(i, str):
            match = tuple(
                p
                for p in parameters_
                if (
                    (isinstance(p, Parameter) and p.name == i)
                    or (isinstance(p, ArrayBox) and p._value.name == i)
                )
            )
            if len(match) == 0:
                return None
            if len(match) == 1:
                match = match[0]
            return match

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
        """Get realization of all child models

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

        Returns
        -------
        model: list
            Realization of the child models, ordered by child index
        """
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

    def update(self):
        """Update internal state or configuration of the model

        The method is only needed to adjust setting or parameters outside of the
        optimization forward path.

        Raises
        ------
        `scarlet.model.UpdateException` if the optimization needs to be interrupted
        """
        pass
