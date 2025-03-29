# Contributing to Macrodata Refinement (MDR)

Thank you for your interest in contributing to Macrodata Refinement! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment](#development-environment)
  - [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Guidelines](#commit-guidelines)
  - [Pull Requests](#pull-requests)
- [Coding Standards](#coding-standards)
  - [Type Safety](#type-safety)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Release Process](#release-process)
- [Communication](#communication)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Development Environment

1. **Fork the repository and clone it locally**:
   ```bash
   git clone https://github.com/yourusername/macrodata-refinement.git
   cd macrodata-refinement
   ```

2. **Set up the development environment**:
   ```bash
   # Using our setup script
   ./scripts/setup_dev_env.sh
   
   # Or manually
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Verify your setup**:
   ```bash
   pytest
   ```

### Project Structure

```
macrodata-refinement/
├── .github/            # GitHub-specific files (workflows, templates)
├── src/                # Source code
│   └── mdr/            # Main package
│       ├── core/       # Core functionality
│       ├── io/         # Input/output operations
│       ├── utils/      # Utility functions
│       ├── api/        # API interfaces
│       └── visualization/ # Visualization components
├── tests/              # Tests
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── fixtures/      # Test fixtures
├── examples/           # Example code
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Development Workflow

### Branching Strategy

- **`main`**: The main branch contains the latest stable code.
- **`dev`**: Development branch for integrating features before release.
- **Feature branches**: Create branches from `dev` for new features (naming: `feature/your-feature-name`).
- **Fix branches**: Create branches from `dev` for bug fixes (naming: `fix/issue-description`).

### Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat: add new feature`
- `fix: correct bug`
- `docs: update documentation`
- `style: formatting changes`
- `refactor: code restructuring without changing functionality`
- `test: add or modify tests`
- `chore: maintenance tasks`

### Pull Requests

1. **Create a new branch** from `dev` for your changes.
2. **Make your changes**, adhering to the coding standards.
3. **Test your changes** with automated tests and manual verification.
4. **Commit your changes** following the commit guidelines.
5. **Push your branch** to your fork.
6. **Create a Pull Request** to the `dev` branch of the main repository.
7. **Describe your changes** in the PR, referencing any relevant issues.
8. **Wait for review** and address any feedback.

## Coding Standards

### Type Safety

MDR strongly emphasizes type safety. All code must include:

1. **Type hints** for all function parameters and return values (following PEP 484).
2. **Type validation** using assertions, especially for floating-point numbers.

Example:
```python
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
```

### Testing

- All new features must include unit tests.
- Tests should be placed in the appropriate directory under `tests/`.
- Run the full test suite before submitting a PR.
- Aim for at least 90% test coverage for new code.

### Documentation

- All public functions, classes, and methods must have docstrings following Google style.
- Update the documentation when changing functionality.
- Add examples for non-trivial features.

## Release Process

1. **Versioning**: We follow [Semantic Versioning](https://semver.org/).
2. **Changelog**: All changes are documented in the CHANGELOG.md file.
3. **Releases**: New releases are created from the `main` branch after merging from `dev`.

## Communication

- **Issues**: Create GitHub issues for bugs, feature requests, or questions.
- **Discussions**: Use GitHub Discussions for more open-ended conversations.
- **Security**: Report security vulnerabilities via [our security policy](SECURITY.md).

---

Thank you for contributing to Macrodata Refinement! Your efforts help improve the library for everyone.