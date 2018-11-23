# -*- coding: utf-8 -*-

from setuptools import setup

import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'numpy',
    'pandas',
    'lalsuite',
    'pycbc'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='elk',
    version=versioneer.get_version(),
    description="""A Python package for managing and interacting with
gravitational waveform catalogues.""",
    long_description=readme + '\n\n' + history,
    author="Daniel Williams",
    author_email='daniel.williams@ligo.org',
    url='https://github.com/transientlunatic/elk',
    packages=['elk'],
    package_dir={'elk': 'elk'},
    entry_points={
        'console_scripts': [
            'elk=elk.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=True,
    keywords='elk',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    cmdclass=versioneer.get_cmdclass()
)
