from __future__ import division

from past.utils import old_div
from builtins import object
from future.utils import with_metaclass

__author__ = "wittawat"

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.signal as sig

"""The MIT License (MIT)

Copyright (c) 2015 Wittawat Jitkrittum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

"""Module containing kernel related classes"""


class Kernel(with_metaclass(ABCMeta, object)):
    """Abstract class for kernels"""

    @abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2"""
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        pass


class KHoPoly(Kernel):
    """Homogeneous polynomial kernel of the form
    (x.dot(y))**d
    """

    def __init__(self, degree):
        assert degree > 0
        self.degree = degree

    def eval(self, X1, X2):
        return np.dot(X1, X2.T) ** self.degree

    def pair_eval(self, X, Y):
        return np.sum(X * Y, 1) ** self.degree

    def __str__(self):
        return "KHoPoly(d=%d)" % self.degree


class KLinear(Kernel):
    def eval(self, X1, X2):
        return np.dot(X1, X2.T)

    def pair_eval(self, X, Y):
        return np.sum(X * Y, 1)

    def __str__(self):
        return "KLinear()"


class KGauss(Kernel):
    def __init__(self, sigma2):
        assert sigma2 > 0, "sigma2 must be > 0"
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1 == d2, "Dimensions of the two inputs must be the same"
        D2 = (
            np.sum(X1**2, 1)[:, np.newaxis]
            - 2 * np.dot(X1, X2.T)
            + np.sum(X2**2, 1)
        )
        K = np.exp(old_div(-D2, self.sigma2))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1 == n2, "Two inputs must have the same number of instances"
        assert d1 == d2, "Two inputs must have the same dimension"
        D2 = np.sum((X - Y) ** 2, 1)
        Kvec = np.exp(old_div(-D2, self.sigma2))
        return Kvec

    def __str__(self):
        return "KGauss(w2=%.3f)" % self.sigma2


class KTriangle(Kernel):
    """
    A triangular kernel defined on 1D. k(x, y) = B_1((x-y)/width) where B_1 is the
    B-spline function of order 1 (i.e., triangular function).
    """

    def __init__(self, width):
        assert width > 0, "width must be > 0"
        self.width = width

    def eval(self, X1, X2):
        """
        Evaluate the triangular kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x 1 numpy array
        X2 : n2 x 1 numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1 == 1, "d1 must be 1"
        assert d2 == 1, "d2 must be 1"
        diff = old_div((X1 - X2.T), self.width)
        K = sig.bspline(diff, 1)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x 1 numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1 == 1, "d1 must be 1"
        assert d2 == 1, "d2 must be 1"
        diff = old_div((X - Y), self.width)
        Kvec = sig.bspline(diff, 1)
        return Kvec

    def __str__(self):
        return "KTriangle(w=%.3f)" % self.width


class Ball(Kernel):
    """
    A Kernel which is 1 inside a ball of radius r and 0 outside
    """
    def __init__(self, r):
        self.r = r

    def eval(self, X1, X2):
        """
        Evaluate the ball kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1 == d2, "Dimensions of the two inputs must be the same"
        D2 = (
            np.sum(X1**2, 1)[:, np.newaxis]
            - 2 * np.dot(X1, X2.T)
            + np.sum(X2**2, 1)
        )
        K = np.zeros(D2.shape)
        K[D2 < self.r**2] = 1
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1 == n2, "Two inputs must have the same number of instances"
        assert d1 == d2, "Two inputs must have the same dimension"
        D2 = np.sum((X - Y) ** 2, 1)
        Kvec = np.zeros(D2.shape)
        Kvec[D2 < self.r**2] = 1
        return Kvec

    def __str__(self):
        return "KBall(r=%.3f)" % self.r
