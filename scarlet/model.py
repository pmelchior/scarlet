from abc import ABC, abstractmethod

from .frame import Frame
from .parameter import Parameter
from .bbox import Box


class Model(ABC):
    """A single component in a blend.

    This class acts as base for building models from parameters.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    name: string
        Name for this model
    parameters: list of `~scarlet.Parameter`
    children: list of `~scarlet.Model`
        Subordinate models.
    bbox: `~scarlet.Box`
        Bounding box of this model
    """

    def __init__(self, frame, parameters=None, children=None, bbox=None, **kwargs):

        if parameters is None:
            parameters = ()
        if hasattr(parameters, "__iter__"):
            for p in parameters:
                assert isinstance(p, Parameter)
            self._parameters = parameters
        else:
            assert isinstance(parameters, Parameter)
            self._parameters = tuple(parameters)
        self.check_parameters()

        if children is None:
            children = ()
        if hasattr(children, "__iter__"):
            for c in children:
                assert isinstance(c, Model)
            self._children = children
        else:
            assert isinstance(children, Model)
            self._children = tuple(children)

        assert isinstance(frame, Frame)
        self._frame = frame
        assert isinstance(bbox, Box)
        self._bbox = bbox

        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def parameters(self):
        """List of parameters, including from the children.
        """
        return self._parameters + tuple(p for c in self.children for p in c._parameters)

    @property
    def children(self):
        return self._children

    @property
    def bbox(self):
        """Hyper-spectral bounding box of this model
        """
        return self._bbox

    @property
    def frame(self):
        """Hyper-spectral characteristics is this model
        """
        return self._frame

    def get_parameter(self, i, *parameters):
        # NOTE: index lookup only works if order is not changed by parameter fixing!
        # check parameters first during optimization
        if i < len(parameters):
            return parameters[i]

        # find them from self (use all even if fixed!)
        # but don't use parameter of children!
        if i < len(self._parameters):
            return self._parameters[i]

        return None

    @abstractmethod
    def get_model(self, *parameters, frame=None):
        """Get the model realization

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

        frame: `~scarlet.frame.Frame`
            Frame to project the model into. If `frame` is `None`
            then the model contained in `self.bbox` is returned.

        Returns
        -------
        model: array
            Realization of the model
        """
        pass

    def get_models_of_children(self, *parameters):
        models = []
        # parameters during optimization
        if len(parameters):
            # count non-fixed own parameters
            i = len(self._parameters)
            for c in self._children:
                j = len(c.parameters)
                models.append(c.get_model(*(parameters[i : i + j])))
                i += j
        else:
            for c in self._children:
                models.append(c.get_model())
        return models

    def check_parameters(self):
        """Check that all parameters have finite elements

        Raises
        ------
        `ArithmeticError` when non-finite elements are present
        """
        for k, p in enumerate(self.parameters):
            if not np.isfinite(p).all():
                msg = "Model {}, Parameter '{}' is not finite:\n{}".format(
                    self.__class__.__name__, p.name, p
                )
                raise ArithmeticError(msg)
