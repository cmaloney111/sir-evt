flu-peak-evt Documentation
===========================

Influenza extreme-peak modeling using EVT and SIR.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Installation
------------

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"

Usage
-----

.. code-block:: bash

   run_experiment examples/sample_influenza.csv --output results/

API Reference
-------------

.. automodule:: flu_peak.data
   :members:
   :undoc-members:

.. automodule:: flu_peak.preprocess
   :members:
   :undoc-members:

.. automodule:: flu_peak.models.sir
   :members:
   :undoc-members:

.. automodule:: flu_peak.models.evt
   :members:
   :undoc-members:

.. automodule:: flu_peak.eval
   :members:
   :undoc-members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
