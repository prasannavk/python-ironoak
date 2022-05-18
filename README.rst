========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-ironoak/badge/?style=flat
    :target: https://python-ironoak.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/prasannavk/python-ironoak/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/prasannavk/python-ironoak/actions

.. |requires| image:: https://requires.io/github/prasannavk/python-ironoak/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/prasannavk/python-ironoak/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/prasannavk/python-ironoak/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/prasannavk/python-ironoak

.. |version| image:: https://img.shields.io/pypi/v/ironoak.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/ironoak

.. |wheel| image:: https://img.shields.io/pypi/wheel/ironoak.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/ironoak

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/ironoak.svg
    :alt: Supported versions
    :target: https://pypi.org/project/ironoak

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/ironoak.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/ironoak

.. |commits-since| image:: https://img.shields.io/github/commits-since/prasannavk/python-ironoak/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/prasannavk/python-ironoak/compare/v0.0.0...master



.. end-badges

A package for a convenient interface to oak cameras.

* Free software: Apache Software License 2.0

Installation
============

::

    pip install ironoak

You can also install the in-development version with::

    pip install https://github.com/prasannavk/python-ironoak/archive/master.zip


Documentation
=============


https://python-ironoak.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
