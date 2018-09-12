# -*- coding: utf-8 -*-
import configparser
import os
from pkg_resources import resource_string


__author__ = """Daniel Williams"""
__email__ = 'daniel.williams@ligo.org'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


default_config = resource_string(__name__, 'elk.conf')

config = configparser.ConfigParser()
# if not config_file:
config.read_string(default_config.decode("utf-8"))
# if config_file:
#     config.read(config_file)


os.environ["LAL_DATA_PATH"] = config.get("lalsuite", "data-path")
