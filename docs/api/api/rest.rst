.. _api_api_rest:

REST API
========

.. automodule:: mdr.api.rest
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``rest`` module provides a REST API for the MDR package, allowing remote access
to data refinement, validation, and transformation capabilities. The API is built
with FastAPI and includes comprehensive documentation using OpenAPI.

API Endpoints
-------------

The REST API provides the following endpoints:

- **/refinement**: Refine data by removing outliers, imputing missing values, and smoothing
- **/validation**: Validate data quality using configurable checks

- **/docs**: Interactive API documentation

Server Configuration
--------------------

.. autoclass:: mdr.api.rest.APIConfig
   :no-index:
   :members:

The ``APIConfig`` class defines the configuration for the REST API server,
including host, port, logging settings, and security options.

Core Endpoint Functions
-----------------------

.. autofunction:: mdr.api.rest.refine_data_endpoint
   :no-index:
.. autofunction:: mdr.api.rest.validate_data_endpoint
   :no-index:



Running the API Server
----------------------

The API server can be run using the included command-line interface:

.. code-block:: bash

    # Run the API server with default settings
    mdr-api

    # Run with custom host and port
    mdr-api --host 0.0.0.0 --port 8080

Or using the Python API:

.. code-block:: python

    from mdr.api.rest import start_api_server
    
    # Start the API server
    start_api_server(host="localhost", port=8000)

Docker Deployment
-----------------

The API server can also be deployed using Docker:

.. code-block:: bash

    # Build the Docker image
    docker build -t mdr-api .
    
    # Run the container
    docker run -p 8000:8000 mdr-api

Or using Docker Compose:

.. code-block:: bash

    # Start the API server using Docker Compose
    docker-compose up mdr-api

API Usage Examples
------------------

Using curl:

.. code-block:: bash

    # Refinement request
    curl -X POST "http://localhost:8000/refinement" \
         -H "Content-Type: application/json" \
         -d '{"data": [1.0, 2.0, null, 4.0, 100.0], "config": {"smoothing_factor": 0.2, "outlier_threshold": 2.5, "imputation_method": "linear", "normalization_type": "minmax"}}'

Using Python requests:

.. code-block:: python

    import requests
    import json
    
    # Define the API URL
    api_url = "http://localhost:8000"
    
    # Define data and config
    data = [1.0, 2.0, None, 4.0, 100.0]
    config = {
        "smoothing_factor": 0.2,
        "outlier_threshold": 2.5,
        "imputation_method": "linear",
        "normalization_type": "minmax"
    }
    
    # Send a refinement request
    response = requests.post(
        f"{api_url}/refinement",
        json={"data": data, "config": config}
    )
    
    # Parse the response
    result = response.json()
    print("Refined data:", result["refined_data"])
