""" Functions for estimating hodograph coherency. """

# pylint: disable=not-an-iterable, missing-function-docstring
import numpy as np
from numba import jit_module


def stacked_amplitude(corrected_gather, amplify_factor=0, abs=True):
    numerator = np.nansum(corrected_gather)
    denominator = 1
    if abs:
        numerator = np.abs(numerator)
    n = max(np.sum(~np.isnan(corrected_gather)), np.int64(1))
    numerator = numerator * ((amplify_factor / np.sqrt(n)) + ((1 - amplify_factor) / n))
    return numerator, denominator


def stacked_amplitude_sum(corrected_gather):
    denominator = 1
    return np.nansum(corrected_gather), denominator


def normalized_stacked_amplitude(corrected_gather):
    return np.abs(np.nansum(corrected_gather)), np.nansum(np.abs(corrected_gather))


def semblance(corrected_gather):
    numerator = (np.nansum(corrected_gather) ** 2) / max(np.sum(~np.isnan(corrected_gather)), 1)
    denominator = np.nansum(corrected_gather ** 2)
    return numerator, denominator


def crosscorrelation(corrected_gather):
    numerator = ((np.nansum(corrected_gather) ** 2) - np.nansum(corrected_gather ** 2)) / 2
    denominator = 1
    return numerator, denominator


def energy_normalized_crosscorrelation(corrected_gather):
    input_enerty =  np.nansum(corrected_gather ** 2)
    output_energy = np.nansum(corrected_gather) ** 2
    numerator = (output_energy - input_enerty) / max(np.sum(~np.isnan(corrected_gather)) - 1, 1)
    denominator = input_enerty
    return numerator, denominator


ALL_FASTMATH_FLAGS  = {'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
jit_module(nopython=True, nogil=True, fastmath=ALL_FASTMATH_FLAGS - {'nnan'})
