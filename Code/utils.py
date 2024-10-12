import re
import torch
from torch import distributions as dists
from torch import Tensor as TT
from typing import Callable, Union
TensFunc2 = Callable[[TT, TT], TT]
TensFunc = Callable[[TT], TT]


def sortfilesby(files, string: str):
    index = []
    for file in files:
        match = re.search(string, file)
        index.append(float(match.group(1)))
    return [x for _, x in sorted(zip(index, files))], sorted(index)


def gen_error(data: Union[TT, list], h_hat: TensFunc2, h_true: TensFunc2,
              summary_func: TensFunc = torch.mean, batch_size=None) -> TT:
    """Calculate the error between the true and estimated functions

    Args:
        data (torch.Tensor|list): Tensor or list of tensors to pass to the functions
        h_hat (Callable(torch.Tensor*|torch.Tensor)): estimated function
        h_true (Callable(torch.Tensor*|torch.Tensor)): true function
        summary_func (Callable(torch.Tensor,torch.Tensor), optional): Summary function to average errors.
                                                                      Defaults to torch.mean.
        batch_size (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    errors = []
    if type(data) is torch.Tensor:
        data = [data]
    if batch_size is None:
        split_data = [[datum] for datum in data]
    else:
        split_data = [torch.split(datum, batch_size) for datum in data]

    for i, temp_data in enumerate(zip(*split_data)):
        y_fake = h_hat(*temp_data)
        y_true = h_true(*temp_data)
        errors.append(torch.abs(y_fake - y_true))
    return summary_func(torch.cat(errors))


def torch_normcdf(x: torch.Tensor) -> torch.Tensor:
    """Calculate the normal CDF

    Args:
        x (torch.Tensor): Values to calculate the CDF for

    Returns:
        torch.Tensor: CDF values
    """
    return 0.5 * (1 + torch.erf(x / (2**0.5)))


def get_true_g(gs_0: list, gs_1: list):
    """Get true g function given the transformation functions

    Args:
        gs_0 (list): Transformation functions for A=0 y_out=gs_0[0](x)+gs_0[1](x)*y_in
        gs_1 (list): Transformation functions for A=1 y_out=gs_1[0](x)+gs_1[1](x)*y_in
    """
    def true_g(y, x):
        return gs_1[1](x)*(y-gs_0[0](x))/gs_0[1](x)+gs_1[0](x)
    return true_g


def get_true_h(hs_0: list, hs_1: list, base_cdf: TensFunc2 = torch_normcdf):
    """Get true h function given the transformation functions

    Args:
        hs_0 (list): Transformation functions for A=0 y_out=hs_0[0](x)+hs_0[1](x)*y_in
        hs_1 (list): Transformation functions for A=1 y_out=hs_1[0](x)+hs_1[1](x)*y_in
        base_cdf (Callable((torch.Tensor,torch.Tensor),torch.Tensor), optional): . Defaults to torch_normcdf.
    """
    def true_h(y0, y1, x):
        # transform y0 and y1
        y0_t = (y0 - hs_0[0](x))/hs_0[1](x)
        y1_t = (y1 - hs_1[0](x))/hs_1[1](x)
        # get cdfs
        cdf0 = base_cdf(y0_t)
        cdf1 = base_cdf(y1_t)
        # get the difference
        return cdf1 - cdf0
    return true_h


def get_true_cqte(hs_0: list, hs_1: list, base_cdf: torch.distributions.Distribution):
    """Get true h function given the transformation functions

    Args:
        hs_0 (list): Transformation functions for A=0 y_out=hs_0[0](x)+hs_0[1](x)*y_in
        hs_1 (list): Transformation functions for A=1 y_out=hs_1[0](x)+hs_1[1](x)*y_in
        base_cdf (Callable((torch.Tensor,torch.Tensor),torch.Tensor), optional): . Defaults to torch_normcdf.
    """
    def true_h(alpha, x):
        # transform y0 and y1
        base_y = base_cdf.icdf(alpha)
        y0_t = (base_y)*hs_0[1](x)+hs_0[0](x)
        y1_t = (base_y)*hs_1[1](x)+hs_1[0](x)

        # get the difference
        return y1_t - y0_t
    return true_h


def cdf_term_gen(gs, y_base_dist=dists.Normal(0, 1)):
    def cdf_freq_0(y, x, *g_args, **g_kwargs):
        return y_base_dist.cdf((y-gs[0](x, *g_args, **g_kwargs)) / gs[1](x, *g_args, **g_kwargs))
    return cdf_freq_0


def icdf_term_gen(gs, y_base_dist=dists.Normal(0, 1)):
    def icdf_freq_0(alpha, x, *g_args, **g_kwargs):
        return y_base_dist.icdf(alpha)*gs[1](x, *g_args, **g_kwargs)+gs[0](x, *g_args, **g_kwargs)
    return icdf_freq_0


def pdf_term_gen(gs, y_base_dist=dists.Normal(0, 1)):
    def pdf_term_freq(alpha, x, *g_args, **g_kwargs):
        input_y = y_base_dist.icdf(alpha)
        return torch.exp(y_base_dist.log_prob(input_y))/gs[1](x, *g_args, **g_kwargs)
    return pdf_term_freq


def all_term_gen(gs, y_base_dist=dists.Normal(0, 1)):
    cdf_term = cdf_term_gen(gs, y_base_dist)
    icdf_term = icdf_term_gen(gs, y_base_dist)
    pdf_term = pdf_term_gen(gs, y_base_dist)

    return (cdf_term, icdf_term, pdf_term)


def torch_nanstd(X: torch.Tensor, dim):
    """A version of torch.std that ignores NaN values

    Args:
        X (torch.Tensor): The tensor to calculate the std of
        dim (int): The dimension to calculate the std over

    Returns:
        torch.Tensor: The standard deviation
    """
    squared_diff = (X - torch.nanmean(X, dim=dim, keepdim=True))**2
    return torch.sqrt(torch.nanmean(squared_diff, dim=dim))


def my_all(X: torch.Tensor, dim=None):
    """Version of torch.all that allows for specifying the dimension to apply the all operation

    Args:
        X (torch.Tensor): The tensor to apply the all operation to
        dim (int, optional): Dimensions to apply all over if `None` applies to every dimension. Defaults to None.

    Returns:
        torch.Tensor: logical tensor of whether all values are true
    """
    if dim is None:
        dim = tuple(range(len(X.shape)))
    return torch.sum(X, dim=dim) == torch.prod(torch.tensor(X.shape)[list(dim)])


def my_any(X: torch.Tensor, dim):
    """Version of torch.any that allows for specifying the dimension to apply the any operation

    Args:
        X (torch.Tensor): The tensor to apply the any operation to
        dim (int): Dimensions to apply any over

    Returns:
        torch.Tensor: logical tensor of whether any values are true
    """
    return torch.sum(X, dim=dim) > 0


def my_allclose(X: torch.Tensor, Y: torch.Tensor, rtol=1e-05, atol=1e-08, dim=None):
    """Version of torch.allclose that allows for specifying the dimension to apply the allclose operation

    Args:
        X (torch.Tensor): The first tensor to compare
        Y (torch.Tensor): The second tensor to compare
        rtol (float, optional): Relative tolerance. Defaults to 1e-05.
        atol (float, optional): Absolute tolerance. Defaults to 1e-08.
        dim (int, optional): Dimensions to apply allclose over. Defaults to None.

    Returns:
        torch.Tensor: logical tensor of whether all values are close
    """
    if dim is None:
        dim = tuple(range(len(X.shape)))
    logic_mat = torch.abs(X-Y) < atol + rtol * torch.abs(Y)
    return my_all(logic_mat, dim=dim)


def torch_nancov(X: torch.Tensor):
    """Calculate the covariance matrix of a tensor while ignoring NaN values

    Args:
        X (torch.Tensor): The tensor to calculate the covariance matrix of

    Returns:
        torch.Tensor: The covariance matrix
    """
    d = X.shape[0]
    nan_bool = torch.isnan(X)
    cov_mat = torch.empty((d, d))
    inds = torch.tril_indices(d, d, -1)
    for ind in inds.T:
        X_sub = X[ind, :]
        nan_sub = nan_bool[ind, :]
        X_sub = X_sub[:, ~my_any(nan_sub, 0)]
        temp_cov = torch.cov(X_sub)
        for i, sub_ind in enumerate(ind):
            cov_mat[sub_ind, ind] = temp_cov[i, :]
    return cov_mat


def get_ci(vec: torch.Tensor, dim: int, verbose=False, na_rm=False) -> torch.Tensor:
    """Get mean and CI for mean from vector

    Args:
        vec (Tensor): The vector of values you want the C.I. for the mean from
        dim (int): The dimension to calculate the mean and CI over
        verbose (bool, optional): Whether or not to print the CI. Defaults to True.
    """
    n = vec.shape[0]
    if na_rm:
        n_samples = torch.sum(~torch.isnan(vec), dim=dim)
        mean = torch.nanmean(vec, dim=dim)
        se = torch_nanstd(vec, dim=dim)/(n_samples**0.5)
    else:
        mean = torch.mean(vec, dim=dim)
        se = torch.std(vec, dim=dim)/(n**0.5)
    ci_up = mean+1.96*se
    ci_low = mean-1.96*se
    # if verbose:
    #     print(f"Our Estimated Expected Power is: {mean:4.3f}")
    #     print(f"With ci({ci_low:4.3f}, {ci_up:4.3f})")
    return torch.stack([mean, ci_low, ci_up], dim=0)


def recursive_tensorize(list_of_lists) -> torch.Tensor:
    """Recursively convert a list of lists to a tensor via iterative implementation of stack on the 0th dimension.
       Similar behaviour to applying torch.tensor to a list of lists
       but now allowing the final layer to be a muli-dimensional Tensor.

    Args:
        list_of_lists (list[list[torch.Tensor]]): A varying depth nested list of lists of tensors to convert

    Returns:
        torch.Tensor: The converted tensor
    """
    if isinstance(list_of_lists, list):
        if isinstance(list_of_lists[0], (list, tuple, torch.Tensor)):
            elems_to_stack = [recursive_tensorize(elem) for elem in list_of_lists]
            return torch.stack(elems_to_stack)
        else:
            return torch.tensor(list_of_lists)
    else:
        return list_of_lists
