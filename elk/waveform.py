"""
This module contains code for wrapping lalsuite functionality
in a more pythonic manner to allow the creation of waveforms.
"""


import pycbc.types.timeseries
from pycbc.waveform import get_td_waveform
# import lal
# import lalsimulation as lalsim
import numpy as np
from .exceptions import LalsuiteError
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import minimize


def inner_product(x, y, sample_f, asd="LIGODesign",
                  fmin=30, phase=0, nfft=512):
    """
    Calculate the inner product of two timeseries.


    Parameters
    ----------
    x, y : np.ndarray
       The two timeseries to calculate the inner product for.
    psd : np.ndarray or str or None
       The ASD to use to calculate the match.
       Defaults to "LIGODesign" which is the LIGO design sensitivity.
    fmin : float
       The minimum frequency to be used to calculate the match.
    phase : float
       The phase shift, in radians, to apply to the second time series.

    """
    if asd == "LIGODesign":
        asd = np.loadtxt("fig1_aligo_sensitivity.txt")
    freqs = np.linspace(0, sample_f/2, int(nfft/2)+1)
    x_f = np.fft.rfft(np.hamming(len(x))*x, n=nfft)
    y_f = np.fft.rfft(np.hamming(len(y))*y, n=nfft) * np.exp(1j * phase)
    asd_interp = interp1d(asd[:, 0], asd[:, -2])
    integrand = ((x_f[freqs > fmin]) * np.conj(y_f[freqs > fmin]))
    integrand /= asd_interp(freqs[freqs > fmin])**2
    integral = simps(integrand, x=freqs[freqs > 30])
    return 4*np.real(integral)


def components_from_total(total_mass, mass_ratio):
    """
    Calculate the component black hole masses from the total mass
    of the system and the mass ratio.
    """
    m1 = total_mass / (mass_ratio + 1)
    m2 = total_mass - m1

    return m1, m2


class FrequencySeries(object):
    """
    A class to represent a frequency series (spectrum)
    """
    pass


class Timeseries(object):
    """
    A class to represent a timeseries from LALSuite in a more
    Python-friendly manner.
    """

    def __init__(self, data, times=None, variance=None):
        """
        Create a Timeseries from a LALSuite timeseries or from data and times.

        Optionally, a variance timeseries can be provided.

        Parameters
        ----------
        data : {array-like, pycbc timeseries}
           An array of data points.
        variance : array-like (optional)
           An array of variances.
        times : array-like
           An array of timestamps
        """

        if isinstance(data, pycbc.types.timeseries.TimeSeries):
            # This looks like a LALSuite timeseries
            self.dt = np.diff(data.sample_times)[0]
            self.times = np.array(data.sample_times)
            self.data = np.array(data.data)
        elif isinstance(times, np.ndarray):
            # This looks like separate times and data
            self.times = np.array(times)
            self.data = np.array(data)
            self.dt = np.diff(self.times)[0]

        if isinstance(variance, np.ndarray):
            self.variance = variance

        self.df = 1./self.dt

    def pycbc(self):
        """
        Return the timeseries as a pycbc timeseries.
        """
        return pycbc.types.TimeSeries(self.data, self.dt)

    def apply_phase_offset(self,
                           phase,
                           nfft=512,
                           window=np.hamming):
        """
        Generate the timeseries of this waveform with a defined phase offset.

        Parameters
        ----------
        phase : float
           The phase offset, in radians, to be introduced into the waveform.
        nfft : int
           The length of the fourier transform. Defaults to 512, and should
           be a power of 2 for speed.
        window : numpy window function
           The windowing function to use for the FFT.

        Returns
        -------
        Shifted timeseries : `numpy.ndarray`
           The phase-shifted timeseries.
        """

        def pow2(x):
            """Find the next power of 2"""
            return 1 if x == 0 else 2**(int(x) - 1).bit_length()

        y = self.data

        nfft = pow2(len(y)+np.abs(phase))
        ik = np.array([2j*np.pi*k for k in range(0, nfft)]) / nfft
        y_f = np.fft.fft(window(len(y))*y, n=nfft) * np.exp(- ik * phase)

        return np.real(np.fft.ifft(y_f, len(y)))


class Waveform(object):
    pass


class NRWaveform(Waveform):
    """
    This class represents a waveform object, and can produce
    either the time-domain or the frequency-domain
    representation of the waveform.
    """

    def __init__(self, data_file, parameters):
        """
        Create the waveform object.

        Parameters
        ----------
        data_file : str
           The filepath to the datafile containing this waveform.
        parameters : dict
           A dictionary of this waveform's parameters.

        """

        self.data_file = data_file

        for key, value in parameters.items():
            setattr(self, key, value)

        self.spins = [self.spin_1x, self.spin_1y, self.spin_1z,
                      self.spin_2x, self.spin_2y, self.spin_2z]

    def __repr__(self):
        return "<NR Waveform {} at q={}>".format(self.tag, self.mass_ratio)

    def minimum_frequency(self, total_mass):

        return self.Mflower / total_mass

    def _match(self, x, y, sample_f=1024, fmin=30, phase=0):
        top = inner_product(x, y, sample_f, phase=phase)
        bottom = np.sqrt(inner_product(x, x, sample_f)
                         * inner_product(y, y, sample_f))
        return np.abs(top / bottom)

    def optim_match(self, x, y, sample_f, fmin=30):
        """
        Calculate the optimal match, maximised over the phase shift of the
        two waveforms.
        """
        def neg_match(phase, x, y, sample_f, fmin):
            # Return the neagtive of the match which we can minimise.
            return - self._match(x, y, sample_f, fmin, phase)

        phase_op = minimize(neg_match, x0=0, args=(x, y, sample_f, fmin))

        return -phase_op['fun'], phase_op['x']

    def timeseries(self,
                   total_mass,
                   sample_rate=4096,
                   f_low=None,
                   distance=1,
                   coa_phase=0,
                   ma=None,
                   t_min=None,
                   t_max=None,
                   f_ref=None,
                   t_align=True):
        """
        Generate the timeseries representation of this waveform.
        """

        if not f_low:
            f_low = self.minimum_frequency(total_mass)
        if not f_ref:
            f_ref = f_low

        mass1, mass2 = components_from_total(total_mass, self.mass_ratio)

        try:
            hp, hx = get_td_waveform(approximant="NR_hdf5",
                                     mass1=mass1,
                                     mass2=mass2,
                                     spin1x=self.spin_1x,
                                     spin1y=self.spin_1y,
                                     spin1z=self.spin_1z,
                                     spin2x=self.spin_2x,
                                     spin2y=self.spin_2y,
                                     spin2z=self.spin_2z,
                                     distance=distance,
                                     coa_phase=coa_phase,
                                     delta_t=1.0 / sample_rate,
                                     f_lower=f_low,
                                     inclination=0,
                                     ma=ma,
                                     f_ref=f_ref,
                                     numrel_data=self.data_file)
            hp = Timeseries(hp)
            hx = Timeseries(hx)

            if t_align:
                # Recenter the waveforms on the maximum strain
                hp.times -= hp.times[np.argmax(np.abs(hp.data - 1j * hx.data))]
                hx.times -= hx.times[np.argmax(np.abs(hp.data - 1j * hx.data))]
            if t_min:
                hp.data, hp.times = hp.data[hp.times > t_min], hp.times[hp.times > t_min]
                hx.data, hx.times = hx.data[hx.times > t_min], hx.times[hx.times > t_min]
            if t_max:
                hp.data, hp.times = hp.data[hp.times < t_max], hp.times[hp.times < t_max]
                hx.data, hx.times = hx.data[hx.times < t_max], hx.times[hx.times < t_max]

            return hp, hx

        except RuntimeError:
            raise LalsuiteError
