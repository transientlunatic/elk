"""
This module contains code for wrapping lalsuite functionality
in a more pythonic manner to allow the creation of waveforms.
"""

from pycbc.waveform import get_td_waveform
# import lal
# import lalsimulation as lalsim
# import numpy as np


def components_from_total(total_mass, mass_ratio):
    """
    Calculate the component black hole masses from the total mass
    of the system and the mass ratio.
    """
    m1 = total_mass / (mass_ratio + 1)
    m2 = total_mass - m1

    return m1, m2


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

    def __repr__(self):
        return "<NR Waveform {} at q={}>".format(self.tag, self.mass_ratio)

    def timeseries(self, total_mass, sample_rate=4096,
                   flow=30, distance=1):
        """
        Generate the timeseries representation of this waveform.
        """

        mass1, mass2 = components_from_total(total_mass, self.mass_ratio)

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
                                 delta_t=1.0 / sample_rate,
                                 f_lower=flow,
                                 numrel_data=self.data_file)

        return hp, hx
