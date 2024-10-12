# %%
import numpy as np
import torch
from torch.nn import functional as F  # noqa: F401
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import splrep, splev
from scipy.optimize import fsolve
from abc import ABC
from typing import Union
from . import kernel
TT = torch.Tensor
zero = torch.tensor([0.])
# %%


class kernel_regressor(ABC):
    """A class to perform simple kernel regression with a specified kernel.
    """
    def __init__(self, kernel: kernel.Kernel, min: float = -torch.inf, max: float = torch.inf) -> None:
        """Initialise the kernel regressor with the given kernel.

        Args:
            kernel (kernel.Kernel): kernel function to use for regression.
            min (float, optional): Minimum value for output of regression. Defaults to -torch.inf.
            max (float, optional): Maximum value for output of regression. Defaults to torch.inf.
        """
        self.kernel = kernel
        self.min = min
        self.max = max

    def fit(self, y: TT, X: TT) -> None:
        """Fit the kernel regressor to the given data.

        Args:
            y (torch.Tensor): y values to fit to.
            X (torch.Tensor): x values to fit to (final dim is dimension of x values)
        """
        self.y = y
        self.X = X

    def get_y_weights(self, X_new: TT) -> TT:
        """Get weights (normalised kernels) for each y value given a new X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to get weights for.

        Returns:
            torch.Tensor: Tensor of weights
        """
        X_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X.numpy()))
        return X_dists/torch.sum(X_dists, dim=1, keepdim=True)

    def predict(self, X_new: TT) -> TT:
        """Predict the y values for a given X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to predict y values for (final dim is dim of x).

        Returns:
            torch.Tensor: Predicted y values for each X_new.
        """
        X_dists = self.get_y_weights(X_new)
        preds = torch.sum(X_dists*self.y, dim=1)
        return torch.clamp(preds, min=self.min, max=self.max)

    predict_proba = predict
    __call__ = predict


class kernel_regressor_numpy(ABC):
    """A class to perform simple kernel regression with a specified kernel.
    """
    def __init__(self, kernel: kernel.Kernel, min: float = -np.inf, max: float = np.inf) -> None:
        """Initialise the kernel regressor with the given kernel.

        Args:
            kernel (kernel.Kernel): kernel function to use for regression.
            min (float, optional): Minimum value for output of regression. Defaults to -torch.inf.
            max (float, optional): Maximum value for output of regression. Defaults to torch.inf.
        """
        self.kernel = kernel
        self.min = min
        self.max = max

    def fit(self, X, y) -> None:
        """Fit the kernel regressor to the given data.

        Args:
            y (torch.Tensor): y values to fit to.
            X (torch.Tensor): x values to fit to (final dim is dimension of x values)
        """
        self.y = y
        self.X = X

    def get_y_weights(self, X_new: TT) -> TT:
        """Get weights (normalised kernels) for each y value given a new X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to get weights for.

        Returns:
            torch.Tensor: Tensor of weights
        """
        X_dists = self.kernel.eval(X_new, self.X)
        return X_dists/np.sum(X_dists, axis=1, keepdims=True)

    def predict(self, X_new: TT) -> TT:
        """Predict the y values for a given X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to predict y values for (final dim is dim of x).

        Returns:
            torch.Tensor: Predicted y values for each X_new.
        """
        X_dists = self.get_y_weights(X_new)
        preds = np.sum(X_dists*self.y, axis=1)
        return np.clip(preds, a_min=self.min, a_max=self.max)

    def predict_proba(self, X_new):
        predictions = self.predict(X_new)
        return np.stack((1-predictions, predictions), axis=-1)
    __call__ = predict


# # Kernel Version ##
class kernel_cdf(ABC):
    """Class for kernel based cdf estimation"""

    def __init__(self, kernel: kernel.Kernel, prop_func=None, supremum=False):
        """Initialise the kernel type as well as the propensity function if necessary.


        Args:
            kernel (kernel.Kernel): The kernel to use for the cdf estimation. Called using the eval method.
            prop_func (Callable, optional): The propensity function. Defaults to None.
            supremum (bool, optional): Whether inverse is done via infimum (default) or supremum. Defaults to False.
        """
        self.kernel = kernel
        self.prop_func = prop_func
        self.supremum = supremum
        self._in_inverse_cdf = False

    def fit(self, y: TT, X: TT):
        """Fit the CDF to the given data.

        Args:
            y (torch.Tensor): y values to fit to.
            X (torch.Tensor): x values to fit to (final dim is dimension of x values)
        """
        self.y = y
        self.X = X
        self.y_sorted, self.sort_indices = torch.sort(self.y)
        self.X_sorted = self.X[self.sort_indices]
        if self.prop_func is not None:
            self.prop_scores = self.prop_func(self.X_sorted)
        else:
            self.prop_scores = torch.ones_like(self.X_sorted[:, 0])-.5

    def get_y_weights(self, X_new: TT) -> TT:
        """Get weights for each y value given a new X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to get weights for.

        Returns:
            torch.Tensor: Tensor of weights
        """
        X_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X_sorted.numpy()))
        # Re-adjust for propensity scores if necessary
        X_dists = X_dists/self.prop_scores
        # Normalise
        X_dists = X_dists/torch.sum(X_dists, dim=1, keepdim=True)
        return X_dists

    def getallcdfs(self, X_new: TT):
        """Get all CDF values and step points for each x value in X_new.

        Args:
            X_new (torch.Tensor): Tensor of new X value to evaluate full CDF at.

        Returns:
            torch.Tensor: CDF values (final dim gives CDF values for each step),
            torch.Tensor: step points in y for these CDF values.
        """
        y_weights = self.get_y_weights(X_new)
        cumul_weights = torch.cumsum(y_weights, dim=-1)
        # A rearranging for the case of supremum which is only relevant for inverse cdf
        if self.supremum and self._in_inverse_cdf:
            cumul_weights = torch.cat((
                torch.zeros_like(cumul_weights[:, 0:1]),
                cumul_weights[:, :cumul_weights.shape[1]-1]), dim=-1)
        # Return weights and the change points they're associated with
        return cumul_weights, self.y_sorted

    def cdf(self, y_new: TT, X_new: TT):
        """Evaluate CDF and give y, X pairs.

        Args:
            y_new (torch.Tensor): Tensor of new y values to evaluate CDF at.
            X_new (torch.Tensor): Tensor of new X values to evaluate CDF at (final dimension is dimension of X).

        Returns:
            torch.Tensor: CDF values for each y_new, X_new pair.
        """
        # X_new: dim ..., X_sorted: dim -1
        y_weights = self.get_y_weights(X_new)
        # y_new: dim ..., y_sorted: dim -1
        return torch.sum(y_weights*(self.y_sorted <= y_new.unsqueeze(-1)), dim=-1)

    def inverse_cdf(self, alpha: Union[float, TT], X_new: TT):
        """Get inverse CDF values for a given alpha and X_new.

        Args:
            alpha (torch.Tensor): Porbability value(s) to get inverse CDF for.
            X_new (torch.Tensor): X values to get inverse CDF for.

        Returns:
            torch.Tensor: Inverse CDF values for each alpha, X_new pair.
        """
        # Set the flag for the altering of getallcdfs to in supremum inverse.
        self._in_inverse_cdf = True
        cdf_vals, _ = self.getallcdfs(X_new)
        if type(alpha) is float:
            alpha = torch.tensor([alpha])
        y_expanded = self.y_sorted.unsqueeze(0).expand(X_new.shape[0], -1)

        self._in_inverse_cdf = False
        if not self.supremum:
            valid_ys = torch.where(cdf_vals >= alpha.unsqueeze(-1),
                                   y_expanded, torch.tensor([torch.inf]))

            out_vals = torch.min(valid_ys, dim=-1)[0]
            # Correct for cases with no valid value which currently output inf
            # Instead output maximum of all ys
            # (This theoretically should happen as the largest y-val should always have eCDF 1).
            return torch.minimum(out_vals, self.y_sorted[-1])
        else:
            valid_ys = torch.where(cdf_vals <= alpha.unsqueeze(-1),
                                   y_expanded, torch.tensor([-torch.inf]))

            out_vals = torch.max(valid_ys, dim=1)[0]
            # Correct for cases with no valid value which currently output inf
            # Instead output maximum of all ys
            # (This theoretically should happen as the smallest y-val should always have eCDF 0).
            return torch.maximum(out_vals, self.y_sorted[0])

    __call__ = cdf


class exact_cdf(ABC):
    def __init__(self, CDF, inverse_CDF=None) -> None:
        """Initialise the exact CCDF with the given CCDF and inverse CCDF functions.

        Args:
            CDF (Callable(torch.Tensor,torch.Tensor)): True CCDF function.
            inverse_CDF (torch.Tensor,torch.Tensor, optional): True inverse CCDF function. Defaults to None.
        """
        self.CDF = CDF
        self.inverse_CDF = inverse_CDF

    def fit(self, y: TT, X: TT = None) -> None:
        """Fit the exact CDF to the given data (only y is used).

        Args:
            y (torch.Tensor): y values to fit to (used for step points in getallcdfs).
            X (torch.Tensor, optional): x values to fit to (not used). Defaults to None
        """
        self.y = y
        self.X = X
        self.y_sorted, self.sort_indices = torch.sort(self.y)

    def cdf(self, y_new: TT, X_new: TT):
        """Evaluate CDF and give y, X pairs.

        Args:
            y_new (torch.Tensor): Tensor of new y values to evaluate CDF at.
            X_new (torch.Tensor): Tensor of new X values to evaluate CDF at (final dimension is dimension of X).

        Returns:
            torch.Tensor: CDF values for each y_new, X_new pair.
        """
        return self.CDF(y_new, X_new)

    def inverse_cdf(self, alpha: Union[float, TT], X_new: TT):
        """Get inverse CDF values for a given alpha and X_new.

        Args:
            alpha (torch.Tensor): Porbability value(s) to get inverse CDF for.
            X_new (torch.Tensor): X values to get inverse CDF for.

        Returns:
            torch.Tensor: Inverse CDF values for each alpha, X_new pair.
        """
        if self.inverse_CDF is None:
            raise ValueError("No inverse CDF function defined.")
        return self.inverse_CDF(alpha, X_new)

    def getallcdfs(self, X_new: TT):
        """Get all CDF values and step points for each x value in X_new.

        Args:
            X_new (torch.Tensor): Tensor of new X value to evaluate full CDF at.

        Returns:
            torch.Tensor: CDF values (final dim gives CDF values for each step),
            torch.Tensor: step points in y for these CDF values.
        """
        return self.CDF(self.y_sorted, X_new.unsqueeze(-2)), self.y_sorted

    __call__ = cdf


class pseudo_ipw(ABC):
    def __init__(self, kernel: kernel.Kernel, prop_func=None, normalisation=None):
        """Initalise the pseudo IPW model with the given kernel and propensity function.

        Args:
            kernel (kernel.Kernel): kernel for use in outer kernel regression.
            prop_func (Callable(torch.Tensor,torch.Tensor), optional): Propensity function (already fitted).
                                                                       Defaults to None.
            normalisation (str, optional): Type of normalisation to use for weights options are:
                                                "None" (default) - Normalised accross all X values
                                                "propensity" - Normalised by propensity scores
                                                "separate" - Normalised by propensity scores for each A value
        """
        self.kernel = kernel
        self.prop_func = prop_func
        self.normalisation = "None" if normalisation is None else normalisation

    def fit(self, y0: TT, X0: TT, y1: TT, X1: TT):
        """Fit the pseudo IPW model to the given data.

        Args:
            y0 (torch.Tensor): y0 values to fit to.
            X0 (torch.Tensor): x0 values to fit to (final dim is dimension of x values).
            y1 (torch.Tensor): y1 values to fit to.
            X1 (torch.Tensor): x1 values to fit to (final dim is dimension of x values).
        """
        self.y0 = y0
        self.X0 = X0
        # Sort y1 and X1 for future use
        self.y1_sorted, self.sort_indices_1 = torch.sort(y1)
        self.X1_sorted = X1[self.sort_indices_1, :]
        # Get propensity scores if necessary
        if self.prop_func is not None:
            self.prop_scores0 = 1-self.prop_func(self.X0)
            self.prop_scores1 = self.prop_func(self.X1_sorted)
        else:  # If no propensity scores then just use 0.5
            self.prop_scores0 = torch.ones_like(self.X0[:, 0])-.5
            self.prop_scores1 = torch.ones_like(self.X1_sorted[:, 0])-.5

    def get_y_weights(self, X_new):
        """Get weights for each y value given a new X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to get weights for.

        Returns:
            torch.Tensor: Tensor of weights
        """
        X0_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X0.numpy()))
        X1_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X1_sorted.numpy()))
        if self.normalisation == "None":
            normaliser_0 = normaliser_1 = (
                torch.sum(X0_dists, dim=1, keepdim=True)
                + torch.sum(X1_dists, dim=1, keepdim=True))
        elif self.normalisation == "propensity":
            normaliser_0 = normaliser_1 = (
                torch.sum(X0_dists/self.prop_scores0, dim=1, keepdim=True)
                + torch.sum(X1_dists/self.prop_scores1, dim=1, keepdim=True))
        elif self.normalisation == "separate":
            normaliser_0 = torch.sum(X0_dists/self.prop_scores0, dim=1, keepdim=True)
            normaliser_1 = torch.sum(X1_dists/self.prop_scores1, dim=1, keepdim=True)

        # Normalise
        X0_dists.div_(normaliser_0)
        X1_dists.div_(normaliser_1)
        return X0_dists, X1_dists

    def get_single_h(self, y0_new, y1_new, X_new):
        """Evaluate h at y0_new, y1_new, X_new triples.

        Args:
            y0_new (torch.Tensor): New y0 value to evaluate h at.
            y1_new (torch.Tensor): New y1 value to evaluate h at.
            X_new (torch.Tensor): New X value to evaluate h at.

        Returns:
            torch.Tensor: h values
        """
        X0_dists, X1_dists = self.get_y_weights(X_new)
        # Get contribution for A=0 samples
        term_0 = torch.sum(X0_dists*(self.y0 <= y0_new.unsqueeze(-1))/self.prop_scores0, dim=1)
        # Get contribution for A=1 samples for at all jumping points (i.e. y1 values)
        term_1 = torch.sum(X1_dists*(self.y1_sorted <= y1_new.unsqueeze(-1))/self.prop_scores1, dim=1)
        # Get value of h at each jumping point
        h = term_1 - term_0
        return h

    def get_all_hs(self, y0_new, X_new):
        """Evaluate h at all y1 step points for each y0_new, X_new pair.

        Args:
            y0_new (torch.Tensor): New y0 value to evaluate h at.
            X_new (torch.Tensor): New X value to evaluate h at.

        Returns:
            torch.Tensor: All h values for each y0, X pair and y1 step point (dim-1 contains y1 step points),
            torch.Tensor: y1 step points.
        """
        X0_dists, X1_dists = self.get_y_weights(X_new)
        # Get contribution fo A=0 samples
        term_0 = torch.sum(X0_dists*(self.y0 <= y0_new.unsqueeze(-1))/self.prop_scores0, dim=1, keepdim=True)
        # Get contribution for A=1 samples for at all jumping points (i.e. y1 values)
        term_1s = torch.cumsum(X1_dists/self.prop_scores1, dim=-1)
        # Get value of h at each jumping point
        hs = term_1s - term_0
        return hs, self.y1_sorted

    def predict(self, y0_new: TT, X_new: TT, sortcheck=True):
        """Give the g value for each y0_new, X_new pair.

        Args:
            y0_new (torch.Tensor): New y0 value to predict g at.
            X_new (torch.Tensor): New X value to predict g at.
            sortcheck (bool, optional): Whether to check if step points are already sorted. Defaults to True.

        Raises:
            ValueError: Errors if step points are not sorted.

        Returns:
            torch.Tensor: g values for each y0_new, X_new pair.
        """
        hs, y1_candidate = self.get_all_hs(y0_new, X_new)
        if sortcheck:
            if not torch.all(y1_candidate == torch.sort(y1_candidate)[0]):
                raise ValueError("y1_candidate is not sorted.")
        # Get y1s which give sufficiently large h/ have sufficiently large CDF
        valid_ys = torch.where(hs >= 0, y1_candidate, torch.tensor([torch.inf]))
        # Find the smallest valid y1
        out_vals = torch.min(valid_ys, dim=1)[0]
        # Correct for cases with no valid value which currently output inf
        # Instead output maximum of all ys
        # (This theoretically should happen as the largest y-val should always have eCDF 1).
        return torch.minimum(out_vals, y1_candidate[-1])


class dr_learner(ABC):
    def __init__(self, kernel: kernel.Kernel, cdf_0: kernel_cdf, cdf_1: kernel_cdf, prop_func=None):
        """Initialise the DR learner with the given kernel and CDFs.

        Args:
            kernel (kernel.Kernel): kernel for use in outer kernel regression.
            cdf_0 (kernel_cdf): Estimated CDF for A=0 already fitted.
            cdf_1 (kernel_cdf): Estimated CDF for A=1 already fitted.
            prop_func (Callable(torch.Tensor, torch.Tensor), optional): Estimated propensity function already fitted.
                                                                        Defaults to None.
        """
        self.kernel = kernel
        self.cdf_0 = cdf_0
        self.cdf_1 = cdf_1
        self.prop_func = prop_func

    def fit(self, y0: TT, X0: TT, y1: TT, X1: TT):
        """Fit the pseudo IPW model to the given data.

        Args:
            y0 (torch.Tensor): y0 values to fit to.
            X0 (torch.Tensor): x0 values to fit to (final dim is dimension of x values).
            y1 (torch.Tensor): y1 values to fit to.
            X1 (torch.Tensor): x1 values to fit to (final dim is dimension of x values).
        """
        self.y1_sorted: TT
        self.y0 = y0
        self.X0 = X0
        self.y1_sorted, self.sort_indices_1 = torch.sort(y1)
        self.X1_sorted = X1[self.sort_indices_1, :]
        # Get propensity scores if necessary
        if self.prop_func is not None:
            self.prop_scores0 = 1-self.prop_func(self.X0)
            self.prop_scores1 = self.prop_func(self.X1_sorted)
        else:  # If no propensity scores then just use 0.5
            self.prop_scores0 = torch.ones_like(self.X0[:, 0])-.5
            self.prop_scores1 = torch.ones_like(self.X1_sorted[:, 0])-.5

    def get_y_weights(self, X_new: TT):
        """Get weights (normalised kernels) for each y value given a new X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to get weights for.

        Returns:
            torch.Tensor: Tensor of weights
        """
        X0_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X0.numpy()))
        X1_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X1_sorted.numpy()))
        normaliser = (
            torch.sum(X0_dists, dim=1, keepdim=True)
            + torch.sum(X1_dists, dim=1, keepdim=True))
        # Normalise
        X0_dists.div_(normaliser)
        X1_dists.div_(normaliser)
        return X0_dists, X1_dists

    def get_single_h(self, y0_new: TT, y1_new: TT, X_new: TT):
        """Evaluate h at y0_new, y1_new, X_new triples.

        Args:
            y0_new (torch.Tensor): New y0 value to evaluate h at.
            y1_new (torch.Tensor): New y1 value to evaluate h at.
            X_new (torch.Tensor): New X value to evaluate h at.

        Returns:
            torch.Tensor: h values
        """
        if not (y0_new.shape == y1_new.shape == X_new.shape[:-1]):
            raise ValueError("""y0_new, y1_new and X_new must have the same dimensions
            (excluding final additional dimension of x_new).""")
        # # Get weights for each fitting sample y given our new sample.
        # X_new: dim 0, X0/1_dists: dim 1.
        X0_dists, X1_dists = self.get_y_weights(X_new)
        # # Get CDFs
        # y0/1_new: dim 0, X0/1: dim 1.
        cdf_vals0 = self.cdf_0.cdf(y0_new.unsqueeze(-1), self.X0)
        cdf_vals01 = self.cdf_0.cdf(y0_new.unsqueeze(-1), self.X1_sorted)
        cdf_vals1 = self.cdf_1.cdf(y1_new.unsqueeze(-1), self.X1_sorted)
        cdf_vals10 = self.cdf_1.cdf(y1_new.unsqueeze(-1), self.X0)
        # # Get inidcators/comparisons
        # y0_new: dim 0, y0: dim 1.
        Z0 = (self.y0 <= y0_new.unsqueeze(-1)).float()
        Z1 = (self.y1_sorted <= y1_new.unsqueeze(-1)).float()

        # # Get final h value
        term_0 = torch.sum(X0_dists*((Z0-cdf_vals0)/self.prop_scores0
                                     + cdf_vals0-cdf_vals10), dim=1)
        term_1 = torch.sum(X1_dists*((Z1-cdf_vals1)/self.prop_scores1
                                     + cdf_vals1-cdf_vals01), dim=1)
        h = term_1-term_0
        return h

    def get_all_hs(self, y0_new: TT, X_new: TT, isotonic=False, check_same=False, slow=False):
        """Get all h values for a given y0_new and X_new.

        Args:
            y0_new (torch.Tensor): new y0 data to predict h values for.
            X_new (torch.Tensor): new X data to predict h values for.
            fast (bool, optional): Whether or not to use fast approach. Defaults to False.
            isotonic (bool, optional): Whether or not to project final output to isotonic vector. Defaults to False.
            check_same (bool, optional): Whether to check if dataset for fitting CDF and DR are the same and adjust.
                                         Defaults to False.
            slow (bool, optional): Whether to use slower alternative approach. Defaults to False.

        Returns:
            torch.Tensor: h values with final dim representing all step points, y1 step values used for h values
            torch.Tensor: y1 step values used for h values
        """
        same = False
        if check_same:
            if torch.all(self.cdf_1.y_sorted == self.y1_sorted) and torch.all(self.cdf_1.X_sorted == self.X1_sorted):
                same = True
        # # Get weights for each fitting sample y given our new sample.
        # X_new: dim ..., X0/1_dists: dim -1.
        X0_dists, X1_dists = self.get_y_weights(X_new)

        # # Get CDFs
        # y0_new: dim ..., X0: dim -1.
        cdf_vals0 = self.cdf_0.cdf(y0_new.unsqueeze(-1), self.X0)
        # y0_new: dim ..., X0: dim -1.
        cdf_vals01 = self.cdf_0.cdf(y0_new.unsqueeze(-1), self.X1_sorted)

        # y0_new in dim ..., y0 in dim -1.
        Z0 = (self.y0 <= y0_new.unsqueeze(-1)).float()
        # # Get contribution of A=0 samples
        # y/X_new: dim ..., empty: dim -1
        # Get 0 Term (depending on y0_new)
        term_0 = torch.sum(X0_dists*((Z0-cdf_vals0)/self.prop_scores0+cdf_vals0),
                           dim=-1, keepdim=True)+torch.sum(X1_dists*cdf_vals01, dim=-1, keepdim=True)

        # ### Term 1 Estimation (depending on all y1) ###
        # X1_sorted:dim 0, y1_steps: dim 1
        all_cdf_vals1, y1_cdf_candidate = self.cdf_1.getallcdfs(self.X1_sorted)
        # X0: dim 0, y1_steps: dim 1
        all_cdf_vals10 = self.cdf_1.getallcdfs(self.X0)[0]
        if not same:
            identity_vec = torch.tensor([0., 1.]).repeat_interleave(
                torch.tensor([self.y1_sorted.shape[0], y1_cdf_candidate.shape[0]]))

            # Set value
            all_y1_candidate, all_sort_indices = torch.sort(torch.cat([self.y1_sorted, y1_cdf_candidate]))
            identity_vec = identity_vec[all_sort_indices]
            # Merging
            # Append 0 to the start of each row
            all_cdf_vals1 = torch.cat([torch.zeros(all_cdf_vals1.shape[0], 1), all_cdf_vals1], dim=1)
            all_cdf_vals10 = torch.cat([torch.zeros(all_cdf_vals10.shape[0], 1), all_cdf_vals10], dim=1)

            all_cdf_vals1_expanded = all_cdf_vals1[:, torch.cumsum(identity_vec, dim=0).int()]
            all_cdf_vals10_expanded = all_cdf_vals10[:, torch.cumsum(identity_vec, dim=0).int()]
        else:
            all_cdf_vals1_expanded = all_cdf_vals1
            all_cdf_vals10_expanded = all_cdf_vals10
            all_y1_candidate = y1_cdf_candidate

        if slow:
            # y1: dim 0, all_y1_candidate: dim 1
            all_Z1s = (self.y1_sorted.unsqueeze(-1) <= all_y1_candidate).float()
            # # Get A=1 samples pseudo-outcome
            # empty: dim 0, y/X1: dim 1, all_y_steps: dim 2
            pseudo_outcome_1 = ((all_Z1s-all_cdf_vals1_expanded)/self.prop_scores1.unsqueeze(1)+all_cdf_vals1_expanded)
            # X_new: ..., y1_candidate: dim1
            term_1 = torch.sum(X1_dists.unsqueeze(-1)*pseudo_outcome_1.unsqueeze(0), dim=-2)
            term_10 = torch.sum(X0_dists.unsqueeze(-1)*all_cdf_vals10_expanded.unsqueeze(0), dim=-2)
            term_1s = term_1+term_10

        else:
            # # Alternative approach
            incidicator_term_1 = torch.cumsum(X1_dists/self.prop_scores1, dim=-1)
            if not same:
                # Append 0 to the start of each row
                incidicator_term_1 = torch.cat([torch.zeros(incidicator_term_1.shape[0], 1), incidicator_term_1], dim=1)
                # Expand out indicator term to match all_y1_candidate
                incidicator_term_1_expanded = incidicator_term_1[:, torch.cumsum(identity_vec == 0, dim=0).int()]
            else:
                incidicator_term_1_expanded = incidicator_term_1
            cdf_pseudo_0 = (1-1/self.prop_scores1.unsqueeze(1))*all_cdf_vals1_expanded
            cdf_pseudo_01 = all_cdf_vals10_expanded
            cdf_term0 = torch.sum(X1_dists.unsqueeze(-1)*cdf_pseudo_0.unsqueeze(0), dim=-2)
            cdf_term01 = torch.sum(X0_dists.unsqueeze(-1)*cdf_pseudo_01.unsqueeze(0), dim=-2)
            term_1s = incidicator_term_1_expanded+cdf_term0+cdf_term01

        hs = term_1s - term_0
        if isotonic:
            temp_h = []
            ir = IsotonicRegression()
            for h_row in hs:
                temp_h.append(torch.tensor(ir.fit_transform(np.arange(h_row.shape[0]), h_row)))
            hs = torch.stack(temp_h, dim=0)
        return hs, all_y1_candidate

    def predict(self, y0_new: TT, X_new: TT, sortcheck=False, linear=False,
                isotonic=True, return_hvals=False, fsolve_kwargs=None, **kwargs):
        """Give the g value for each y0_new, X_new pair.

        Args:
            y0_new (torch.Tensor): New y0 value to predict g at.
            X_new (torch.Tensor): New X value to predict g at.
            sortcheck (bool, optional): Whether to check if step points are already sorted. Defaults to True.
            linear (bool, optional): Whether to linearly interpolate between step points. Defaults to False.
            return_hvals (bool, optional): Whether to return h values as well. Defaults to False.
            isotonic (bool, optional): Whether to project h values to isotonic vector. Defaults to True.
            fsolve_kwargs (dict, optional): Arguments to pass to scipy.optimize.fsolve in `linear=True`.
                                            Defaults to None.
            **kwargs: Additional arguments to pass to get_all_hs.
        Raises:
            ValueError: Errors if step points are not sorted.

        Returns:
            torch.Tensor: g values for each y0_new, X_new pair.
        """
        hs, y1_candidate = self.get_all_hs(y0_new, X_new, isotonic=isotonic, **kwargs)
        if sortcheck:
            if not torch.all(y1_candidate == torch.sort(y1_candidate)[0]):
                raise ValueError("y1_candidate is not sorted.")
        # If h kept discrete
        if not linear:
            # Get y1s which give sufficiently large h/ have sufficiently large CDF
            valid_ys = torch.where(hs >= 0, y1_candidate, torch.tensor([torch.inf]))
            # Find the smallest valid y1
            out_vals = torch.min(valid_ys, dim=-1)[0]
            # Correct for cases with no valid value which currently output inf
            # Instead output maximum of all ys
            # (This theoretically should happen as the largest y-val should always have eCDF 1).
            out_ys = torch.minimum(out_vals, y1_candidate[-1])
            if return_hvals:
                return out_ys, self.get_single_h(y0_new, out_ys, X_new)
            else:
                return out_ys
        # If h made continuous via linear interpolation
        else:
            if fsolve_kwargs is None:
                fsolve_kwargs = {}
            if len(hs.shape) > 2:
                raise ValueError("Continuous prediction doesn't support additional batching dimensions.")
            results = []
            h_out = []
            # Iterate over each y_0 sample and associated h values
            for h_sub in hs:
                # Define function to optimise over y_1 as linear interpolation of h values
                def h_opt(y_opt):
                    return np.interp(y_opt, y1_candidate, h_sub)
                # Solve for y_1
                # Ensure start point comfortably inside interpolation region
                start_point = y1_candidate[y1_candidate.shape[0]//2]
                sol, infodict, ier, mesg = fsolve(h_opt, start_point, full_output=True, **fsolve_kwargs)
                results.append(torch.tensor(sol))
                h_out.append(torch.tensor(infodict['fvec']))

            # Combine values and clamp to ensure no strange behaviour outside interpolation region
            y_out = torch.clamp(torch.cat(results, dim=0), y1_candidate[0], y1_candidate[-1])
            h_out = torch.cat(h_out, dim=0)
            if return_hvals:
                return y_out, h_out
            else:
                return y_out


class conditional_pdf(ABC):
    def __init__(self, density_regression, cdf: kernel_cdf):
        self.density_regression = density_regression
        self.cdf = cdf

    @staticmethod
    def cond_density_kernel(x, h=1):
        return 1/h * np.exp(-x**2/h**2/2)

    def nested_outcome_func(self, quantiles, X, Y):
        return self.cond_density_kernel(Y-quantiles)

    def fit(self, y: TT, X: TT, alpha: float):
        self.y = y
        self.X = X
        self.alpha = torch.tensor(alpha)
        quantile_vals = self.cdf.inverse_cdf(self.alpha, self.X)
        outputs = self.nested_outcome_func(quantile_vals, self.X, self.y)
        self.density_regression.fit(self.X, outputs)

    def predict(self, X_new: TT):
        return self.density_regression.predict(X_new)


class exact_conditional_pdf(ABC):
    def __init__(self, conditional_pdf):
        self.conditional_pdf = conditional_pdf

    def fit(self, y: TT, X: TT, alpha: float):
        self.alpha = torch.tensor(alpha)

    def predict(self, X_new: TT):
        return self.conditional_pdf(self.alpha, X_new)


class dr_learner_quantile_static(ABC):

    def __init__(self, kernel: kernel.Kernel, cdf_0: kernel_cdf, cdf_1: kernel_cdf, pdf_0, pdf_1, prop_func=None):
        """Initialise the DR learner with the given kernel and CDFs.

        Args:
            kernel (kernel.Kernel): kernel for use in outer kernel regression.
            cdf_0 (kernel_cdf): Estimated CDF for A=0 already fitted.
            cdf_1 (kernel_cdf): Estimated CDF for A=1 already fitted.
            prop_func (Callable(torch.Tensor, torch.Tensor), optional): Estimated propensity function already fitted.
                                                                        Defaults to None.
        """
        self.kernel = kernel
        self.cdf_0 = cdf_0
        self.cdf_1 = cdf_1
        self.pdf_0 = pdf_0
        self.pdf_1 = pdf_1
        self.prop_func = prop_func

    def fit(self, y0: TT, X0: TT, y1: TT, X1: TT, alpha: float):
        """Fit the pseudo IPW model to the given data.

        Args:
            y0 (torch.Tensor): y0 values to fit to.
            X0 (torch.Tensor): x0 values to fit to (final dim is dimension of x values).
            y1 (torch.Tensor): y1 values to fit to.
            X1 (torch.Tensor): x1 values to fit to (final dim is dimension of x values).
        """
        self.y1_sorted: TT
        self.y0 = y0
        self.X0 = X0
        self.y1_sorted, self.sort_indices_1 = torch.sort(y1)
        self.X1_sorted = X1[self.sort_indices_1, :]
        # Get propensity scores if necessary
        if self.prop_func is not None:
            self.prop_scores0 = 1-self.prop_func(self.X0)
            self.prop_scores1 = self.prop_func(self.X1_sorted)
        else:  # If no propensity scores then just use 0.5
            self.prop_scores0 = torch.ones_like(self.X0[:, 0])-.5
            self.prop_scores1 = torch.ones_like(self.X1_sorted[:, 0])-.5
        self.alpha = torch.tensor(alpha)

        # y0/1_new: dim 0, X0/1: dim 1.
        quantile_vals0 = self.cdf_0.inverse_cdf(self.alpha, self.X0)
        quantile_vals01 = self.cdf_0.inverse_cdf(self.alpha, self.X1_sorted)
        quantile_vals1 = self.cdf_1.inverse_cdf(self.alpha, self.X1_sorted)
        quantile_vals10 = self.cdf_1.inverse_cdf(self.alpha, self.X0)
        # y0_new: dim 0, y0: dim 1.
        Z0 = self.alpha-(self.y0 <= quantile_vals0).float()
        Z1 = self.alpha-(self.y1_sorted <= quantile_vals1).float()
        self.pdf_vals_0 = self.pdf_0.predict(self.X0)
        self.pdf_vals_1 = self.pdf_1.predict(self.X1_sorted)

        # # Get final h value
        self.pseudo_0 = Z0/(self.prop_scores0*self.pdf_vals_0) + quantile_vals0-quantile_vals10
        self.pseudo_1 = Z1/(self.prop_scores1*self.pdf_vals_1) + quantile_vals1-quantile_vals01

    def get_y_weights(self, X_new: TT):
        """Get weights (normalised kernels) for each y value given a new X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to get weights for.

        Returns:
            torch.Tensor: Tensor of weights
        """
        X0_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X0.numpy()))
        X1_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X1_sorted.numpy()))
        normaliser = (
            torch.sum(X0_dists, dim=1, keepdim=True)
            + torch.sum(X1_dists, dim=1, keepdim=True))
        # Normalise
        X0_dists.div_(normaliser)
        X1_dists.div_(normaliser)
        return X0_dists, X1_dists

    def predict(self, X_new: TT):
        """Evaluate h at y0_new, y1_new, X_new triples.

        Args:
            y0_new (torch.Tensor): New y0 value to evaluate h at.
            y1_new (torch.Tensor): New y1 value to evaluate h at.
            X_new (torch.Tensor): New X value to evaluate h at.

        Returns:
            torch.Tensor: h values
        """
        # # Get weights for each fitting sample y given our new sample.
        # X_new: dim 0, X0/1_dists: dim 1.
        X0_dists, X1_dists = self.get_y_weights(X_new)
        # # Get CDFs

        h = torch.sum(X0_dists*self.pseudo_0, dim=-1) + torch.sum(X1_dists*self.pseudo_1, dim=-1)
        return h


class dr_learner_quantile(ABC):

    def __init__(self, kernel: kernel.Kernel, cdf_0: kernel_cdf, cdf_1: kernel_cdf, pdf_0, pdf_1, prop_func=None):
        """Initialise the DR learner with the given kernel and CDFs.

        Args:
            kernel (kernel.Kernel): kernel for use in outer kernel regression.
            cdf_0 (kernel_cdf): Estimated CDF for A=0 already fitted.
            cdf_1 (kernel_cdf): Estimated CDF for A=1 already fitted.
            prop_func (Callable(torch.Tensor, torch.Tensor), optional): Estimated propensity function already fitted.
                                                                        Defaults to None.
        """
        self.kernel = kernel
        self.cdf_0 = cdf_0
        self.cdf_1 = cdf_1
        self.pdf_0 = pdf_0
        self.pdf_1 = pdf_1
        self.prop_func = prop_func

    @staticmethod
    def exp_kernel_generator(x, h=1):
        return 1/h * np.exp(-x**2/h**2/2)

    def nested_outcome_func(self, quantiles, X, Y):
        return self.cond_density_kernel(Y-quantiles)

    def fit(self, y0: TT, X0: TT, y1: TT, X1: TT):
        """Fit the pseudo IPW model to the given data.

        Args:
            y0 (torch.Tensor): y0 values to fit to.
            X0 (torch.Tensor): x0 values to fit to (final dim is dimension of x values).
            y1 (torch.Tensor): y1 values to fit to.
            X1 (torch.Tensor): x1 values to fit to (final dim is dimension of x values).
        """
        self.y1_sorted: TT
        self.y0 = y0
        self.X0 = X0
        self.y1_sorted, self.sort_indices_1 = torch.sort(y1)
        self.X1_sorted = X1[self.sort_indices_1, :]
        # Get propensity scores if necessary
        if self.prop_func is not None:
            self.prop_scores0 = 1-self.prop_func(self.X0)
            self.prop_scores1 = self.prop_func(self.X1_sorted)
        else:  # If no propensity scores then just use 0.5
            self.prop_scores0 = torch.ones_like(self.X0[:, 0])-.5
            self.prop_scores1 = torch.ones_like(self.X1_sorted[:, 0])-.5

    def get_y_weights(self, X_new: TT):
        """Get weights (normalised kernels) for each y value given a new X value.

        Args:
            X_new (torch.Tensor): Tensor of new X values to get weights for.

        Returns:
            torch.Tensor: Tensor of weights
        """
        X0_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X0.numpy()))
        X1_dists = torch.tensor(self.kernel.eval(X_new.numpy(), self.X1_sorted.numpy()))
        normaliser = (
            torch.sum(X0_dists, dim=1, keepdim=True)
            + torch.sum(X1_dists, dim=1, keepdim=True))
        # Normalise
        X0_dists.div_(normaliser)
        X1_dists.div_(normaliser)
        return X0_dists, X1_dists

    def predict(self, alpha: TT, X_new: TT):
        """Evaluate h at y0_new, y1_new, X_new triples.

        Args:
            y0_new (torch.Tensor): New y0 value to evaluate h at.
            y1_new (torch.Tensor): New y1 value to evaluate h at.
            X_new (torch.Tensor): New X value to evaluate h at.

        Returns:
            torch.Tensor: h values
        """
        # # Get weights for each fitting sample y given our new sample.
        # X_new: dim 0, X0/1_dists: dim 1.
        X0_dists, X1_dists = self.get_y_weights(X_new)
        # # Get CDFs
        # y0/1_new: dim 0, X0/1: dim 1.
        quantile_vals0 = self.cdf_0.inverse_cdf(alpha.unsqueeze(-1), self.X0)
        quantile_vals01 = self.cdf_0.inverse_cdf(alpha.unsqueeze(-1), self.X1_sorted)
        quantile_vals1 = self.cdf_1.inverse_cdf(alpha.unsqueeze(-1), self.X1_sorted)
        quantile_vals10 = self.cdf_1.inverse_cdf(alpha.unsqueeze(-1), self.X0)

        # # Get inidcators/comparisons
        # y0_new: dim 0, y0: dim 1.
        Z0 = alpha.unsqueeze(-1)-(self.y0 >= quantile_vals0).float()
        Z1 = alpha.unsqueeze(-1)-(self.y1_sorted >= quantile_vals1).float()

        # # Get final h value
        term_0 = torch.sum(X0_dists*(Z0/(self.prop_scores0*self.pdf_vals_0)
                                     + quantile_vals0-quantile_vals10), dim=1)
        term_1 = torch.sum(X1_dists*(Z1/(self.prop_scores1*self.pdf_vals_1)
                                     + quantile_vals1-quantile_vals01), dim=1)
        h = term_1-term_0
        return h


class separate_learner(ABC):
    """A class to perform separate kernel regression for each treatment group."""
    def __init__(self, cdf_0: kernel_cdf, cdf_1: kernel_cdf):
        """Initialise the separate learner with the given CDFs.

        Args:
            cdf_0 (kernel_cdf): Estimated CDF for A=0 (already fitted).
            cdf_1 (kernel_cdf): Estimated CDF for A=1 (already fitted).
        """
        self.cdf_0 = cdf_0
        self.cdf_1 = cdf_1

    def fit(self):
        pass

    def get_single_h(self, y_0: TT, y_1: TT, X: TT):
        """Evaluate h at y0_new, y1_new, X_new triples.

        Args:
            y_0 (torch.Tensor): New y0 value to evaluate h at.
            y_1 (torch.Tensor): New y1 value to evaluate h at.
            X (torch.Tensor): New X value to evaluate h at.

        Returns:
            torch.Tensor: h values
        """
        return self.cdf_1.cdf(y_1, X) - self.cdf_0.cdf(y_0, X)

    def get_all_hs(self, y_0: TT, X: TT, **kwargs):
        """Get all h values for a given y0_new and X_new.

        Args:
            y0_new (torch.Tensor): new y0 data to predict h values for.
            X_new (torch.Tensor): new X data to predict h values for.

        Returns:
            torch.Tensor: h values with final dim representing all step points, y1 step values used for h values
            torch.Tensor: y1 step values used for h values
        """
        all_cdfs_1, y_1_candidate = self.cdf_1.getallcdfs(X)
        cdf_0 = self.cdf_0.cdf(y_0, X)
        return all_cdfs_1 - cdf_0.unsqueeze(-1), y_1_candidate

    def predict(self, y_0: TT, X: TT, **kwargs):
        """Give the g value for each y0_new, X_new pair.

        Args:
            y0_new (torch.Tensor): New y0 value to predict g at.
            X_new (torch.Tensor): New X value to predict g at.
            sortcheck (bool, optional): Whether to check if step points are already sorted. Defaults to True.
            linear (bool, optional): Whether to linearly interpolate between step points. Defaults to False.
            return_hvals (bool, optional): Whether to return h values as well. Defaults to False.
            fsolve_kwargs (dict, optional): Arguments to pass to scipy.optimize.fsolve in `linear=True`.
                                            Defaults to None.
            **kwargs: Additional arguments to pass to get_all_hs.
        Raises:
            ValueError: Errors if step points are not sorted.

        Returns:
            torch.Tensor: g values for each y0_new, X_new pair.
        """
        return dr_learner.predict(self, y_0, X, linear=False, isotonic=False)


class separate_quantile_learner(ABC):
    """A class to perform separate kernel regression for each treatment group."""
    def __init__(self, quantile_0: kernel_cdf, quantile_1: kernel_cdf):
        """Initialise the separate learner with the given CDFs.

        Args:
            cdf_0 (kernel_cdf): Estimated CDF for A=0 (already fitted).
            cdf_1 (kernel_cdf): Estimated CDF for A=1 (already fitted).
        """
        self.quantile_0 = quantile_0
        self.quantile_1 = quantile_1

    def fit(self):
        pass

    def predict(self, alpha: TT, X: TT):
        """Evaluate h at y0_new, y1_new, X_new triples.

        Args:
            y_0 (torch.Tensor): New y0 value to evaluate h at.
            y_1 (torch.Tensor): New y1 value to evaluate h at.
            X (torch.Tensor): New X value to evaluate h at.

        Returns:
            torch.Tensor: h values
        """
        return self.quantile_0.predict(alpha, X) - self.quantile_0.predict(alpha, X)


# Create spline regressor using scipy.interpolate splrep and splev
class spline_regressor(ABC):
    """Spline regression class"""
    def __init__(self, **spline_kwargs):
        """Initialise the spline regressor with the given spline kwargs.
        """
        self.spline_kwargs = spline_kwargs

    def fit(self, y: TT, X: TT):
        """Fit the spline regressor to the given data.

        Args:
            y (torch.Tensor): y values to fit to.
            X (torch.Tensor): x values to fit to.
        """
        self.X = X
        self.y = y
        self.X_sorted, self.sort_indices = torch.sort(X.squeeze())
        self.y_sorted = y[self.sort_indices]
        self.tks = splrep(self.X_sorted.numpy(), self.y_sorted.numpy(), **self.spline_kwargs)

    def predict(self, new_X: TT):
        """Predict the y values for a given X value.

        Args:
            new_X (torch.Tensor): Tensor of new X values to predict y values for.

        Returns:
            torch.Tensor: Predicted y values for each X_new.
        """
        return torch.tensor(splev(new_X.squeeze().numpy(), self.tks))

    __call__ = predict
