"""
Elk projection
--------------

This module contains code for projecting timeseries into alternative bases for dimensionality and order reduction.
"""

from scipy.interpolate import LinearNDInterpolator
import numpy as np
class ProperBasis(object):
    """
    Construct and operate a proper orthogonal decomposition of a set of waveforms.
    """
    def __init__(self, locs,
                 timeseries, bases=1, interpolator=LinearNDInterpolator):
        """
        Construct a properly orthogonalised basis representation of a set of training timeseries.

        Parameters
        ----------
        locs : `ndarray` 
           An array of the parameter space coordinates for the 
           training timeseries.
        timeseries : `ndarray`
           An array of training timeseries.
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
        >>> basis = ProperBasis(locs = waveforms.waveforms, 
                    timeseries = data.T, 
                    bases = 2)
        >>> basis(0.5)
        array([ 5.34402492e-20, -2.85781392e-20, -1.04712322e-19,  2.51580757e-20,
       -5.75478506e-21, -3.52799281e-20,  1.41297522e-20, -2.98353891e-20,
        1.94312058e-20, -7.85569366e-21])
        """
        
        self.timeseries = np.array(timeseries)
        self.locs_array = np.array([np.array(list(loc.values())) for loc in locs])
        self.locs_array = self.locs_array[:,:2]
        
        self.locs = locs
        self.basis, self.coeffs = self.proper_decomposition()
        
        self.basis = self.basis[:,:bases]
        self.coeffs = self.coeffs[:,:bases]

        self._interpolator = interpolator(np.atleast_2d(self.locs_array), self.coeffs.T)
        
    def __call__(self, p):
        """
        Produce a waveform
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
        coefficiencts = basis.T.dot(self.timeseries.T)

        return basis, coefficiencts
    
    def determine_mapping(self):
        pass
