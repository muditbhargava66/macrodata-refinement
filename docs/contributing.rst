.. _contributing:

Contributing Guide
================

Thank you for your interest in contributing to Macrodata Refinement (MDR)! This guide 
provides information on how to contribute to the project.

Getting Started
-------------

1. **Fork the repository**:
   
   Fork the repository on GitHub and clone it locally:
   
   .. code-block:: bash
   
      git clone https://github.com/yourusername/macrodata-refinement.git
      cd macrodata-refinement

2. **Set up the development environment**:
   
   .. code-block:: bash
   
      # Using the setup script
      ./scripts/setup_dev_env.sh
      
      # Or manually
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"
      pre-commit install

3. **Verify your setup**:
   
   .. code-block:: bash
   
      pytest

Development Workflow
------------------

Branching Strategy
~~~~~~~~~~~~~~~~

- **main**: The main branch contains the latest stable code.
- **dev**: Development branch for integrating features before release.
- **Feature branches**: Create branches from ``dev`` for new features (naming: ``feature/your-feature-name``).
- **Fix branches**: Create branches from ``dev`` for bug fixes (naming: ``fix/issue-description``).

Commit Guidelines
~~~~~~~~~~~~~~~

We follow the `Conventional Commits <https://www.conventionalcommits.org/>`_ specification:

- ``feat: add new feature``
- ``fix: correct bug``
- ``docs: update documentation``
- ``style: formatting changes``
- ``refactor: code restructuring without changing functionality``
- ``test: add or modify tests``
- ``chore: maintenance tasks``

Pull Request Process
~~~~~~~~~~~~~~~~~~

1. **Create a new branch** from ``dev`` for your changes.
2. **Make your changes**, adhering to the coding standards.
3. **Test your changes** with automated tests and manual verification.
4. **Commit your changes** following the commit guidelines.
5. **Push your branch** to your fork.
6. **Create a Pull Request** to the ``dev`` branch of the main repository.
7. **Describe your changes** in the PR, referencing any relevant issues.
8. **Wait for review** and address any feedback.

Coding Standards
--------------

Type Safety
~~~~~~~~~~

MDR strongly emphasizes type safety. All code must include:

1. **Type hints** for all function parameters and return values (following PEP 484).
2. **Type validation** using assertions, especially for floating-point numbers.

Example:

.. code-block:: python

   def calculate_metric(value: float, factor: float = 1.0) -> float:
       """
       Calculate a metric based on the input value and factor.
       
       Args:
           value: Input value
           factor: Scaling factor (default: 1.0)
           
       Returns:
           Calculated metric
       """
       assert isinstance(value, float), "value must be a floating-point number"
       assert isinstance(factor, float), "factor must be a floating-point number"
       
       return value * factor

Testing
~~~~~~

- All new features must include unit tests.
- Tests should be placed in the appropriate directory under ``tests/``.
- Run the full test suite before submitting a PR.
- Aim for at least 90% test coverage for new code.

Documentation
~~~~~~~~~~~

- All public functions, classes, and methods must have docstrings following Google style.
- Update the documentation when changing functionality.
- Add examples for non-trivial features.

Building Documentation
--------------------

To build the documentation locally:

1. Install the documentation dependencies:
   
   .. code-block:: bash
   
      pip install -e ".[docs]"

2. Build the documentation:
   
   .. code-block:: bash
   
      cd docs
      make html

3. Open the documentation in your browser:
   
   .. code-block:: bash
   
      # On macOS
      open _build/html/index.html
      
      # On Linux
      xdg-open _build/html/index.html
      
      # On Windows
      start _build/html/index.html

Release Process
-------------

1. **Versioning**: We follow `Semantic Versioning <https://semver.org/>`_.
2. **Changelog**: All changes are documented in the CHANGELOG.md file.
3. **Releases**: New releases are created from the ``main`` branch after merging from ``dev``.

Communication
-----------

- **Issues**: Create GitHub issues for bugs, feature requests, or questions.
- **Discussions**: Use GitHub Discussions for more open-ended conversations.
- **Security**: Report security vulnerabilities via our security policy.

Thank you for contributing to Macrodata Refinement!
