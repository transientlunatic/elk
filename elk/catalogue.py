import glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lalsimulation import SimInspiralNRWaveformGetSpinsFromHDF5File as get_spin

from elk import config
from .exceptions import FileTypeError
from .waveform import NRWaveform

# -


class Catalogue(object):
    pass


class NRCatalogue(Catalogue):
    """
    This class represents an NR waveform catalogue.
    """

    def __init__(self, origin="GeorgiaTech", ftype="hdf5"):
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

        self.table, self.waveforms = self.assemble_from_directory()
        self.waveforms = np.array(self.waveforms, dtype=object)

    def assemble_from_directory(self):
        """
        Assemble a catalogue by analysing the contents of a directory.
        """

        nr_files = glob.glob(self.data_path + "/*." + self.file_suffix)

        columns = ["tag", "mass_ratio",
                   "spin_1x", "spin_1y", "spin_1z",
                   "spin_2x", "spin_2y", "spin_2z"]

        self.parameters = ["mass_ratio", "spin_1x", "spin_1y", "spin_1z",
                           "spin_2x", "spin_2y", "spin_2z"]

        df = pd.DataFrame(columns=columns)

        waveforms = []

        for nrfile in nr_files:

            data = h5py.File(nrfile, 'r')

            eta = data.attrs['eta']
            mass_ratio = -(2*eta + np.sqrt(1-4*eta) - 1) / (2*eta)

            # This doesn't actually matter, but lalsim *requires* it...
            total_mass = 100
            s1x, s1y, s1z, s2x, s2y, s2z = get_spin(0, total_mass, nrfile)

            pars = [nrfile.split("/")[-1].split(".")[0],
                    mass_ratio, s1x, s1y, s1z, s2x, s2y, s2z]

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

        query_table, query_waveforms = self.query(
            "spin_1x == 0 & spin_1y == 0 & spin_1z == 0"
            + "& spin_2x == 0 & spin_2y == 0 & spin_2z == 0")

        return query_table, query_waveforms

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

        query_table = self.table.query(expression, inplace=False)
        query_waveforms = self.waveforms[
            np.array(query_table.index, dtype=int)]

        return query_table, query_waveforms

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
        """

        distances = self.distances(point)

        closest = np.argmin(distances)

        return distances[closest], self.waveforms[closest]

    def coverage_plot(self, figsize=(9, 9)):
        """
        Plot an n-dimensional corner plot to illustrate the
        parameter space coverage of this catalogue.
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

                if j == len(self.parameters):
                    ax.set_xlabel(parameter)
                else:
                    ax.set_xticks([])

                if i == 0:
                    ax.set_ylabel(j_parameter)
                else:
                    ax.set_yticks([])

        f.tight_layout()
        f.savefig("test_corner.png")
