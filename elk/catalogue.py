import glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lalsimulation import SimInspiralNRWaveformGetSpinsFromHDF5File as get_spin

from . import config
from .exceptions import FileTypeError, LalsuiteError
from .waveform import NRWaveform, Timeseries

from pycbc.waveform import get_td_waveform
# -


class Catalogue(object):

    def components_from_total(self, total_mass, mass_ratio):
        """
        Calculate the component black hole masses from the total mass
        of the system and the mass ratio.
        """
        m1 = total_mass / (mass_ratio + 1)
        m2 = total_mass - m1

        return m1, m2


class PPCatalogue(Catalogue):
    """
    This class represents a (continuous) catalogue
    based off an analytical waveform approximant.
    """

    solMass = 1.988e30  # kg
    parsec = 3.0857e16  # m

    def __init__(self, approximant, total_mass=20, fmin=30.):
        """
        Assemble the catalogue.

        Parameters
        ----------
        approximant : str
           The name of the waveform approximant.

        """

        self.approximant = approximant
        self.total_mass = total_mass
        self.fmin = fmin

    def waveform(self, p, time_range, distance=1.0, coa_phase=0):
        """
        Generate a single waveform from the catalogue.
        """

        delta_t = (time_range[1] - time_range[0]) / time_range[2]
        mass1, mass2 = self.components_from_total(self.total_mass,
                                                  p['mass ratio'])

        hp, hx = get_td_waveform(approximant=self.approximant,
                                 mass1=mass1,
                                 mass2=mass2,
                                 spin1x=p['spin 1x'],
                                 spin1y=p['spin 1y'],
                                 spin1z=p['spin 1z'],
                                 spin2x=p['spin 2x'],
                                 spin2y=p['spin 2y'],
                                 spin2z=p['spin 2z'],
                                 distance=distance,
                                 delta_t=delta_t/1e4,
                                 coa_phase=coa_phase,
                                 f_lower=self.fmin)

        hp = Timeseries(hp)
        hx = Timeseries(hx)

        # Recenter the waveforms on the maximum strain
        hp.times -= hp.times[np.argmax(np.abs(hp.data))]
        hx.times -= hx.times[np.argmax(np.abs(hx.data))]

        tix = (time_range[0] < hp.times*1e4) & (hp.times*1e4 < time_range[1])

        hp.times = hp.times[tix]
        hx.times = hx.times[tix]
        hp.data = hp.data[tix]
        hx.data = hx.data[tix]

        return hp, hx


class NRCatalogue(Catalogue):
    """
    This class represents an NR waveform catalogue.
    """

    def __init__(self, origin="GeorgiaTech", ftype="hdf5",
                 table=None, waveforms=None):
        """
        Assemble a catalogue of numerical relativity waveforms.

        Parameters
        ----------
        origin: str {"GeorgiaTech", "RIT", "SXS"}
           The source of the waveforms to build the catalogue from.
           At present this list is limited to a small number of NR
           simulation sources.

        ftype: str
           The file type containing the waveforms.
        """

        if ftype == "hdf5":
            self.file_suffix = "h5"
        else:
            raise FileTypeError

        self.data_path = config.get("catalogues", origin)

        if isinstance(waveforms, np.ndarray) \
           and isinstance(table, pd.DataFrame):
            self.table = table
            self.waveforms = waveforms
        else:
            self.table, self.waveforms = self.assemble_from_directory()
            self.waveforms = np.array(self.waveforms, dtype=object)

        self.parameters = ["mass_ratio", "spin_1x", "spin_1y", "spin_1z",
                           "spin_2x", "spin_2y", "spin_2z"]

        self.data_parameters = {0: "time",
                                1: "mass ratio",
                                2: "spin 1x",
                                3: "spin 1y",
                                4: "spin 1z",
                                5: "spin 2x",
                                6: "spin 2y",
                                7: "spin 2z",
                                8: "h+",
                                9: "hx"}
        self.c_ind = {j: i for i, j in self.data_parameters.items()}

    def assemble_from_directory(self):
        """
        Assemble a catalogue by analysing the contents of a directory.
        """

        nr_files = glob.glob(self.data_path + "/*." + self.file_suffix)

        columns = ["tag", "mass_ratio",
                   "spin_1x", "spin_1y", "spin_1z",
                   "spin_2x", "spin_2y", "spin_2z", "Mflower"]

        df = pd.DataFrame(columns=columns)
        waveforms = []

        for nrfile in nr_files:

            data = h5py.File(nrfile, 'r')

            try:
                eta = data.attrs['eta']
            except KeyError:
                print("Problem with waveform {}".format(nrfile))

            Mflower = data.attrs['f_lower_at_1MSUN']
            mass_ratio = -(2*eta + np.sqrt(1-4*eta) - 1) / (2*eta)

            # This doesn't actually matter?, but lalsim *requires* it...
            total_mass = 60
            s1x, s1y, s1z, s2x, s2y, s2z = get_spin(0, total_mass, str(nrfile))

            pars = [nrfile.split("/")[-1].split(".")[0],
                    mass_ratio, s1x, s1y, s1z, s2x, s2y, s2z,
                    Mflower]

            df.loc[-1] = pars
            df.index = df.index + 1
            df = df.sort_index()

            parameters = dict(zip(columns, pars))

            waveforms.append(NRWaveform(nrfile, parameters))

        return df, waveforms

    def spin_free(self):
        """
        Filter the catalogue to only show non-spinning waveforms.
        """

        new_cat = self.query(
            "spin_1x == 0 & spin_1y == 0 & spin_1z == 0"
            " & spin_2x == 0 & spin_2y == 0 & spin_2z == 0")

        return new_cat

    def query(self, expression):
        """
        Query the catalogue to return waveforms which match specific
        conditions.

        Parameters
        ----------
        expression: str
           A query expression, for example 's1x == 0' to find all
           the waveforms where the s1x component is zero.
        """

        print(expression)
        query_table = self.table.query(expression, inplace=False)
        print(query_table)
        query_waveforms = self.waveforms[
            np.array(query_table.index, dtype=int)]

        new_cat = NRCatalogue(origin="GeorgiaTech", ftype="hdf5",
                              table=query_table,
                              waveforms=query_waveforms)
        return new_cat

    def distances(self, point, metric=np.ones(7)):
        """
        Calculate the distance between a point and each
        of the waveforms in the catalogue.
        """
        point = np.array(point, dtype=np.float)

        parameters = np.array(
            self.table[self.parameters],
            dtype=np.float
        )
        return np.sqrt(np.sum((parameters - point)**2, axis=1))

    def find_closest(self, point):
        """
        Find the closest waveform to a given point, and return it.

        Parameters
        ----------
        point : `np.ndarray` or `list`
           An arbitrary point in parameter space from which
           to search for the nearest waveform.
        """

        distances = self.distances(point)

        closest = np.argmin(distances)

        return distances[closest], self.waveforms[closest]

    def coverage_plot(self, figsize=(9, 9)):
        """
        Plot an n-dimensional corner plot to illustrate the
        parameter space coverage of this catalogue.

        Parameters
        ----------
        figsize : tuple
           The size of the figure to be produced.

        Returns
        -------
        figure: `matplotlib.figure.Figure`#
           The figure object containing the corner plot.
        """

        f = plt.figure(figsize=figsize)

        # produce an n x n grid of subplots
        gs = gridspec.GridSpec(len(self.parameters), len(self.parameters),
                               wspace=0.0, hspace=0.0)

        for i, parameter in enumerate(self.parameters):
            for j, j_parameter in enumerate(self.parameters):

                if i > j:
                    # Don't produce a plot for combinations above the
                    # diagonal.
                    continue
                else:
                    # Produce a subplot for combinations on or below
                    # the diagonal
                    ax = plt.subplot(gs[j, i])

                if i == j:
                    # This is the on-diagonal case, which we'll just skip for
                    # now. Ideally will want to insert a histogram here.
                    ax.hist(self.table[parameter])
                else:
                    # Produce a scatter plot of the waveforms for this
                    # combination
                    ax.scatter(self.table[parameter], self.table[j_parameter])

                if j == len(self.parameters) - 1:
                    ax.set_xlabel(parameter.replace("_", " "))
                else:
                    ax.set_xticks([])

                if i == 0:
                    ax.set_ylabel(j_parameter.replace("_", " "))
                else:
                    ax.set_yticks([])

        f.tight_layout()
        return f

    def minimum_frequency(self, total_mass):

        freqs = []
        for waveform in self.waveforms:
            freqs.append(waveform.minimum_frequency(total_mass))

        return np.min(freqs)

    def create_training_data(self, total_mass, fmin=30,
                             sample_rate=4096, distance=1, tmax=0.005):
        """
        Produce an array of data suitable for use as training data
        using the waveforms in this catalogue.

        Parameters
        ----------
        total_mass : float
           The total mass, in solar masses, at which the
           waveforms should be produced.
        fmin : float
           The minimum frequency whcih should be included
           in the wavefom.
        sample_rate : float
           The sample rate at which the data should be generated.
        distance : float
           The distance, in megaparsecs, at which the source should
           be located compared to the observer.

        Returns
        -------
        training_data : `numpy.ndarray`
           An array of the training data, with the appropriate
           'y' values in the final columns.
        """

        big_export = np.zeros((1, 10))

        gen_sample = 4096
        skip = int(gen_sample/sample_rate)  # this is nasty
        for waveform in self.waveforms:
            try:
                hp, hx = waveform.timeseries(total_mass, gen_sample,
                                             fmin, distance)
            except LalsuiteError:
                print(
                    "There was an error producing a waveform for {}"
                    .format(waveform.tag))
                continue

            ixs = hp.times < tmax

            export = np.ones((len(hp.data[ixs][::skip]), 10))

            export[:, 0] = hp.times[ixs][::skip]
            export[:, 1] = waveform.mass_ratio
            export[:, [2, 3, 4, 5, 6, 7]] *= waveform.spins
            export[:, 8] = hp.data[ixs][::skip]
            export[:, 9] = hx.data[ixs][::skip]

            big_export = np.vstack([big_export, export])

        return big_export[1:, :]
