mille-feuille
=============

.. image:: https://github.com/aidancrilly/mille-feuille/actions/workflows/run_tests.yaml/badge.svg
   :target: https://github.com/aidancrilly/mille-feuille/actions/workflows/run_tests.yaml
   :alt: CI status

**mille-feuille** acts as an orchestrator when running sampling, learning and
optimisation loops against expensive MPI-parallelised HPC codes.  For
optimisation, it is a thin wrapper on top of `BoTorch
<https://botorch.org/>`_, providing the necessary interface between
simulators, surrogates and optimisers.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
