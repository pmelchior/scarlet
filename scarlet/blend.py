import autograd.numpy as np
from autograd import grad

from .component import ComponentTree, BlendFlag

import logging

logger = logging.getLogger("scarlet.blend")


class Blend(ComponentTree):
    """The blended scene

    The class represents a scene as collection of components, internally as a
    `~scarlet.component.ComponentTree`, and provides the functions to fit it
    to data.

    Attributes
    ----------
    mse: list
        Array of mean squared errors in each iteration
    """

    def __init__(self, sources, observations):
        """Constructor

        Form a blended scene from a collection of `~scarlet.component.Component`s

        Parameters
        ----------
        sources: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
            Intitialized components or sources to fit to the observations
        observations: a `scarlet.Observation` instance or a list thereof
            Data package(s) to fit
        """
        ComponentTree.__init__(self, sources)

        try:
            iter(observations)
        except TypeError:
            observations = (observations,)
        self.observations = observations

        n_params = 2 * self.K
        self._grad = grad(self._loss, tuple(range(n_params)))
        self.mse = []

    @property
    def it(self):
        """Number of iterations run in the `fit` method
        """
        return len(self.mse)

    @property
    def converged(self):
        """Whether the full Blend has converged within e_rel.

        For convergence tests of individual components, check its `flags` member.
        """
        for c in self.components:
            if (c.flags & (BlendFlag.SED_NOT_CONVERGED | BlendFlag.MORPH_NOT_CONVERGED)).value > 0:
                return False
        return True

    def fit(self, max_iter=200, e_rel=1e-2, approximate_L=False):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float
            Relative error for convergence of each component.
        approximate_L: bool
            Whether or not to use a rough approximation of the
            Lipschitz constants
        """
        t = np.zeros((9, max_iter))
        count = 0
        for step in range(max_iter):
            # Back propagate the gradients
            import time
            t[0,count] = time.clock()
            self._backward()
            t[1, count] = time.clock()

            # Calculate the Lipschitz constants,
            # which are needed to determine the step size for each component
            self._set_lipschitz(approximate_L)
            t[2, count] = time.clock()
            # Take the next gradient step for each component
            for c in self.components:
                c.L_sed = self.L_sed
                t[3, count] = time.clock()
                c.L_morph = self.L_morph
                t[4, count] = time.clock()
                c.backward_prior()
                t[5, count] = time.clock()
                if not c.fix_sed:
                    c._sed = c._sed - c.step_sed * c.sed_grad
                    t[6, count] = time.clock()
                    c.sed_grad = 0
                if not c.fix_morph:
                    c._morph = c._morph - c.step_morph * c.morph_grad
                    t[7, count] = time.clock()
                    c.morph_grad = 0

            # Call the update functions for all of the sources
            self.update()
            t[8, count] = time.clock()
            count+=1

            if self._check_convergence(e_rel):
                break
        import matplotlib.pyplot as plt
        plt.plot(t[1, :]-t[0,:], label = 'self._backward')
        plt.plot(t[2, :]-t[1, :], label='set Lipschitz')
        plt.plot(t[3, :]-t[2, :], label='L_sed')
        plt.plot(t[4, :]-t[3, :], label='Lmorph')
        plt.plot(t[5, :]-t[4, :], label='backward prior')
        plt.plot(t[6, :]-t[5, :], label='grad step sed')
        plt.plot(t[7, :]-t[6, :], label='grad step morhp')
        plt.plot(t[8, :]-t[7, :], label='self.update')
        plt.yscale('log')
        plt.legend()

        plt.xlabel('iterations')
        plt.ylabel('time ()')

        plt.savefig('Ndim'+str(count))
        plt.show()


    def _backward(self):
        """Backpropagate the gradients for the seds and morphs
        """
        seds = [c.sed for c in self.components]
        morphs = [c.morph for c in self.components]
        parameters = seds + morphs
        # This calculates the partial derivatives wrt
        # all the seds and morphologies
        gradients = self._grad(*parameters)
        sed_gradients = gradients[:self.K]
        morph_gradients = gradients[self.K:]
        # set the sed and morphology gradients for each source
        for k,c in enumerate(self.components):
            c.sed_grad = sed_gradients[k]
            c.morph_grad = morph_gradients[k]

    def _loss(self, *parameters):
        """Loss function for autograd

        This method combines the seds and morphologies
        into a model that is used to calculate the loss
        function and update the gradient for each
        parameter
        """
        # Unpack the seds and morphologies
        seds = parameters[:self.K]
        morphs = parameters[self.K:]

        model = self.get_model(seds, morphs)
        # Caculate the total loss function from all of the observations
        total_loss = 0
        for observation in self.observations:
            total_loss = total_loss + observation.get_loss(model)
        self.mse.append(total_loss._value)
        return total_loss

    def _check_convergence(self, e_rel):
        """Check to see if all of the components have converged

        The `flag` property of each component is updated to reflect
        whether or not its SED and morphology have converged.

        Parameters
        ----------
        e_rel: float
            Relative error for convergence of each component.

        Returns
        -------
        converged: bool
            Whether or not all of the components have converged.
        """
        e_rel2 = e_rel ** 2
        if self.it > 1:
            converged = True
            for component in self.components:
                # sed convergence
                diff2 = ((component._last_sed - component.sed) ** 2).sum()
                if diff2 <= e_rel2 * (component.sed ** 2).sum():
                    component.flags &= ~BlendFlag.SED_NOT_CONVERGED
                else:
                    component.flags |= BlendFlag.SED_NOT_CONVERGED
                    converged = False
                # morph convergence
                diff2 = ((component._last_morph - component.morph) ** 2).sum()
                if diff2 <= e_rel2 * (component.morph ** 2).sum():
                    component.flags &= ~BlendFlag.MORPH_NOT_CONVERGED
                else:
                    component.flags |= BlendFlag.MORPH_NOT_CONVERGED
                    converged = False
        else:
            converged = False

        # Store a copy of each SED and morphology for the
        # convergence check in the next iteration
        for c in self.components:
            c._last_sed = c.sed.copy()
            c._last_morph = c.morph.copy()

        return converged

    def _set_lipschitz(self, approximate_L):
        """Update the Lipschitz constants from the observations
        """
        if approximate_L:
            # spectral norm of S*S^T or A^T*A is of order the mean amplitude
            # of their elements times the number of components
            LA, LS = 0, 0
            for c in self.components:
                LS += (c.sed ** 2).sum().item()
                LA += (c.morph ** 2).sum().item()

            # if loss has increased from last iteration, increase L
            # TODO: this is insufficient if the estimation above is miles off!
            # Then: escalate the increase
            if self.it > 1 and self.mse[-1] > self.mse[-2]:
                LS *= 2
                LA *= 2
        else:
            # This is still an approximation, but a less crude (albeit slower) one
            C, Ny, Nx = self.frame.shape
            seds = np.zeros((self.K, C), dtype=self.components[0].sed.dtype)
            morphs = np.zeros((self.K, Ny, Nx), dtype=self.components[0].morph.dtype)
            for k, component in enumerate(self.components):
                seds[k] = component.sed
                morphs[k] = component.morph
            _S = morphs.reshape(morphs.shape[0], -1)
            _SST = _S.dot(_S.T)
            _ATA = seds.T.dot(seds)
            # Lipschitz constant for A
            LA = np.real(np.linalg.eigvals(_SST).max())
            # Lipschitz constant for S
            LS = np.real(np.linalg.eigvals(_ATA).max())

        LA *= len(self.observations)
        LS *= len(self.observations)

        self.L_sed = LA
        self.L_morph = LS
