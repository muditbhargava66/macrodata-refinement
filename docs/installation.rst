.. _installation:

Installation
============

This section covers how to install Macrodata Refinement (MDR) in different environments.

Prerequisites
-------------

MDR requires Python 3.10 or newer. It also depends on the following key packages:

* numpy
* pandas
* matplotlib
* scikit-learn (optional, for advanced features)

From PyPI (Recommended)
-----------------------

The easiest way to install MDR is using pip:

.. code-block:: bash

    pip install macrodata-refinement

For development features:

.. code-block:: bash

    pip install "macrodata-refinement[dev]"

The development version includes additional dependencies for testing, documentation, and development.

From Source
-----------

To install from source, first clone the repository:

.. code-block:: bash

    git clone https://github.com/muditbhargava66/macrodata-refinement.git
    cd macrodata-refinement
    pip install -e .

This installs MDR in development mode, so changes to the source code will be reflected immediately.

Using Docker
------------

MDR provides a Docker image for easy deployment, especially for the API:

.. code-block:: bash

    docker-compose up mdr-api

This will start the MDR API server on the default port. You can configure the port and other settings in the `docker-compose.yml` file.

Verifying Installation
----------------------

To verify that MDR is installed correctly, you can run:

.. code-block:: python

    import mdr
    print(mdr.__version__)

Or run a simple example:

.. code-block:: python

    from mdr.core.refinement import RefinementConfig
    
    # Create a configuration object
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    
    print("MDR installed successfully!")

Troubleshooting
---------------

Common installation issues and their solutions:

1. **Missing dependencies**: If you encounter a `ModuleNotFoundError`, make sure all required packages are installed.

   .. code-block:: bash

       pip install -r requirements.txt

2. **Version conflicts**: If you have version conflicts with existing packages, consider using a virtual environment:

   .. code-block:: bash

       python -m venv mdr-env
       source mdr-env/bin/activate  # On Windows: mdr-env\Scripts\activate
       pip install macrodata-refinement

3. **Docker issues**: If you encounter issues with Docker, ensure Docker and Docker Compose are installed and that the required ports are available.
