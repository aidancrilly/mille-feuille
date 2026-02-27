Installation
============

Pip (latest development head)
------------------------------

.. code-block:: bash

   pip install git+https://github.com/aidancrilly/mille-feuille.git

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/aidancrilly/mille-feuille.git
   cd mille-feuille
   pip install -e .[dev]
   python dev_fetch_deps.py

Requires **Python ≥ 3.11**.  Core dependencies (``botorch``, ``gpytorch``,
``numpy``, ``scipy``, ``h5py``) are pulled in automatically.

Building the documentation locally
-----------------------------------

Install the extra documentation dependencies:

.. code-block:: bash

   pip install -e .[docs]

Then build the HTML docs:

.. code-block:: bash

   cd docs
   make html

The output will be in ``docs/_build/html/``.
