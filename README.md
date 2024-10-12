# QuantileComparator
This is the code for the paper "Conditional Quantile Comparator: A Quantile Alternative to CATE" by Josh Givens, Henry Reeve, Song Liu, and Katarzyna Reluga. The paper is published in proceedings for NeurIPS 2024 and can be found at [Placeholder].

As a quick summary, this method creates a function to transform values from one conditional distirbution to a value of the equivalent quantile in another conditional distribution.

# Folder contents
## `CDTE`
This contains the code for the CDTE method presented in Kallus and Oprescu 2023. The paper and original code for this can be found at:
https://proceedings.mlr.press/v206/kallus23a.html

## `Code`
This contains all the code for the method.  All estimators are defined in `nonparamcdf.py` with the kernels themselves implemented in `kernel.py` which is adapted from https://github.com/wittawatj/kernel-gof/ the original licence for this code can be found in the file as well. `utils.py` contains general helper functions for the code.
## `Experiments`
This contains notebooks for all the experiments in the paper.  ColonExample.ipynb contains the code for the colon cancer example, EmploymentExample.ipynb contains the code for the employment example, and `SimulatedExperiment.ipynb` contains the code for all the simulated examples. All experimental results are saved in the `Test_Results` folder.
## `Plots`
This contains all the plots for the paper.
## `RealData`
The contains the colon data and employment data used in the paper as well as the R code to retrieve and process the colon data.

# Datasets
## Employment data
The Employment data is taken from https://www.journals.uchicago.edu/doi/suppl/10.1086/687522/suppl_file/12062data.zip, the supplementary material for https://www.journals.uchicago.edu/doi/10.1086/687522
## Colon Cancer Data
The colon cancer data is found in the `survival` package in R.  The data is loaded with the following code:
```{r}
library(survival)
data(colon)
```
Code to convert this data from long into wide format and then save to csv can be found in `RealData/data_read.R`.

# Code Usage
The main classes are `dr_learner`, `pseudo_ipw`, and `separate_learner` which can all be used to estimate h at given points using the `get_single_h` method, evaulate at all y_1 step points using the `get_all_hs` method and estimate $g^*$ using the `predict` method. Each must be initially fit using the `fit` method with the data used to estimate the function.

There are also `kernel_cdf` and `exact_cdf` methods for estimating the cdf with `cdf` for giving evaluation of the cdf at specified $y,x$ points and `getallcdfs` for evaluating the cdf at all $y$ points for specified $x$ points. Both have to be fit using the `fit` method with the data used to estimate the cdf.

Finally there is the `kernel_regressor` class to perform standard regression with the `fit` method and evaluate the regression at specified points with the `predict` method.