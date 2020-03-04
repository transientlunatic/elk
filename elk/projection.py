"""
Elk projection
--------------

This module contains code for projecting timeseries into alternative bases for dimensionality and order reduction.
"""

from scipy.interpolate import LinearNDInterpolator, interp1d
import numpy as np
import json

class ProperBasis(object):
    """
    Construct and operate a proper orthogonal decomposition of a set of waveforms.
    """
    def __init__(self, locs,
                 timeseries, times=None, bases=1, interpolator=LinearNDInterpolator):
        """Construct a properly orthogonalised basis representation of a set of training timeseries.

        .. warning:: This class is very much still under development, and you should use it at your peril (well, with caution that its interface will change).

        Parameters
        ----------
        locs : `ndarray` 
           An array of the parameter space coordinates for the 
           training timeseries.
        timeseries : `ndarray`
           An array of training timeseries.
        times: `ndarray`
           The times corresponding to each basis vector.
        bases : int
           The number of bases which should be used to produce the new basis.
        interpolator : function
           The function to interpolate between points in the reduced space.

        Examples
        --------
        >>> waveforms = PPCatalogue(approximant="IMRPhenomPv2", 
                                    total_mass=20, 
                                    fmin=30, 
                                    waveforms = [
                {"mass ratio": q, 
                 "spin 1x": 0, "spin 1y": 0, 
                 "spin 1z": 0, "spin 2x": 0, 
                 "spin 2y": 0, "spin 2z": 0}
                for q in np.linspace(0.1, 1.0, 10)
            ])
        >>> time_range = [-0.05, 0.01, 10]
        >>> data = np.array([waveforms.waveform(p=waveform_p, 
                                    time_range=time_range)[0].data 
                 for waveform_p in waveforms.waveforms])
        >>> locs = np.array([np.array(list(waveform.values())) for waveform in waveforms.waveforms])
        >>> basis = ProperBasis(locs = locs[:,0], 
                    timeseries = data, 
                    bases = 20)
        >>> basis(0.5)
        array([-1.89929950e-20,  1.66736553e-20, -7.28592335e-21,  1.14458442e-21,
       -3.13984587e-21,  1.13517145e-20, -1.98235910e-20,  2.25379863e-20,
        -1.60227815e-20,  5.26962113e-22])
        """
        
        self.timeseries = np.array(timeseries)
        self.locs_array = locs #
        
        self.locs = locs
        self.basis, self.coeffs = self.proper_decomposition()
        
        self.basis = self.basis#[:,:bases]
        self.coeffs = self.coeffs#[:,:bases]

        self.times = times

        # TODO Fix this up properly
        if self.locs_array.ndim == 1:
            self._interpolator = interp1d(self.locs_array, self.coeffs)
        else:
            print(self.locs_array.shape)
            print(self.coeffs.shape)
            self._interpolator = interpolator(self.locs_array, self.coeffs.T)
        
    def __call__(self, p):
        """Produce a waveform

        Parameters
        ----------
        p : `ndarray`-like
           An array describing the parameters for the new waveform.
        """
        
        return np.dot(self.basis, self.interpolated_coefficients(p).T)
        
    def interpolated_coefficients(self, p):
        """
        Determine the coordinates in the reduced space.
        """
        return self._interpolator(p)
        
    def proper_decomposition(self):
        """
        Calculate the proper orthogonal decomposition of a set of waveforms.
        """

        u, s, vh = np.linalg.svd(self.timeseries.T, full_matrices=False)
        basis = u
        coefficiencts = basis.dot(self.timeseries)

        return basis, coefficiencts

    def save(self, filename):
        """Save the basis to a machine-readable file."""
        data = dict(vectors = self.basis.tolist(), abscissa = self.times.tolist(), coefficients = self.coeffs.T.tolist(), locations = self.locs.tolist())

        with open("{}.json".format(filename), "w") as fp:
            json.dump(data, fp)
