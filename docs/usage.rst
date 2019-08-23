=====
Usage
=====

Configuration
=============

Before you start using elk you'll need to tell it where the data you want to work with is located.
The easiest way to do this is to make a configuration file in your home directory; elk will look for configuration files in the following locations, in order of precedence::

  ./elk.conf
  ~/.config/elk/elk.conf
  ~/.elk
  /etc/elk


The configuration file should contain a list of catalogues and the location of the directory where the hd5 data files are stored in this format: ::

  [catalogues]
  GeorgiaTech = /home/daniel/scratch/data/lvcnr-lfs/GeorgiaTech
  SXS = /home/daniel/scratch/data/lvcnr-lfs/SXS/res5
  RIT = GeorgiaTech = /home/daniel/scratch/data/lvcnr-lfs/RIT

In addition your configuration file should contain the location of the lal-data directory on your machine, for example: ::

   [lalsuite]
   data-path = /home/daniel/data/gravitational-waves/lal-data

A simple use-case
=================

To use Elk in a project you just need to import it::

    import elk

  You can then load-up a catalogue from the set which are defined in the configuration file, for example, the GeorgiaTech catalogue: ::

   catalogue = elk.catalogue.NRCatalogue(origin="GeorgiaTech")  

    
You can then plot the coverage of the catalogue over parameter space: ::

  catalogue.coverage_plot()

or access the table of waveforms: ::

  print(catalogue.table)


Using a phenomenological model with elk
=======================================

Sometimes it can be helpful to treat a phenomenological model in the same way as a catalogue of NR waveforms.

TODO *Need to add documentation for this*
