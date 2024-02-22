""" Functions for estimating hodograph coherency. """

# pylint: disable=not-an-iterable, missing-function-docstring
import numpy as np
from numba import jit_module


def stacked_amplitude(hodograph, amplify_factor=0, abs=True):
    numerator = np.nansum(hodograph)
    denominator = 1
    if abs:
        numerator = np.abs(numerator)
    n = max(np.sum(~np.isnan(hodograph)), np.int64(1))
    numerator = numerator * ((amplify_factor / np.sqrt(n)) + ((1 - amplify_factor) / n))
    return numerator, denominator


def stacked_amplitude_sum(hodograph):
    denominator = 1
    return np.nansum(hodograph), denominator


def normalized_stacked_amplitude(hodograph):
    return np.abs(np.nansum(hodograph)), np.nansum(np.abs(hodograph))


def semblance(hodograph):
    numerator = (np.nansum(hodograph) ** 2) / max(np.sum(~np.isnan(hodograph)), 1)
    denominator = np.nansum(hodograph ** 2)
    return numerator, denominator


def crosscorrelation(hodograph):
    numerator = ((np.nansum(hodograph) ** 2) - np.nansum(hodograph ** 2)) / 2
    denominator = 1
    return numerator, denominator


def energy_normalized_crosscorrelation(hodograph):
    input_energy =  np.nansum(hodograph ** 2)
    output_energy = np.nansum(hodograph) ** 2
    numerator = (output_energy - input_energy) / max(np.sum(~np.isnan(hodograph)) - 1, 1)
    denominator = input_energy
    return numerator, denominator


ALL_FASTMATH_FLAGS  = {'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
jit_module(nopython=True, nogil=True, fastmath=ALL_FASTMATH_FLAGS - {'nnan'})
