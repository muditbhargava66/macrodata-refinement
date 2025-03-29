#!/bin/bash
#
# Setup script for Macrodata Refinement (MDR) development environment.
#
# This script:
# 1. Creates and activates a virtual environment
# 2. Installs development dependencies
# 3. Sets up pre-commit hooks
# 4. Configures git (optional)
#

set -e  # Exit on error

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Function to print section header
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning message
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go up one level to the project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for Python 3.10+
print_header "Checking Python version"
PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+')
PYTHON_VERSION_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_VERSION_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

if [ "$PYTHON_VERSION_MAJOR" -lt 3 ] || ([ "$PYTHON_VERSION_MAJOR" -eq 3 ] && [ "$PYTHON_VERSION_MINOR" -lt 10 ]); then
    print_error "Python 3.10 or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

print_success "Found Python $PYTHON_VERSION"

# Create and activate virtual environment
print_header "Setting up virtual environment"

VENV_DIR="${PROJECT_ROOT}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
else
    print_warning "Virtual environment already exists at $VENV_DIR"
fi

# Detect shell and create appropriate activation command
if [ -n "$BASH_VERSION" ]; then
    ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
elif [ -n "$ZSH_VERSION" ]; then
    ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
elif [ -n "$FISH_VERSION" ]; then
    ACTIVATE_CMD="source ${VENV_DIR}/bin/activate.fish"
else
    ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
fi

# Activate virtual environment
echo "Activating virtual environment..."
eval "$ACTIVATE_CMD"

# Verify that virtual environment is activated
if [[ "$VIRTUAL_ENV" != "$VENV_DIR" ]]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

print_success "Virtual environment activated"

# Install dependencies
print_header "Installing dependencies"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Check if the package is already installed in development mode
if pip list | grep -q "macrodata-refinement"; then
    print_warning "Package already installed. Checking for updates..."
    pip install -e ".[dev]" --upgrade
else
    echo "Installing package in development mode..."
    pip install -e ".[dev]"
fi

print_success "Dependencies installed"

# Set up pre-commit hooks
print_header "Setting up pre-commit hooks"

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Check if .pre-commit-config.yaml exists
PRE_COMMIT_CONFIG="${PROJECT_ROOT}/.pre-commit-config.yaml"
if [ ! -f "$PRE_COMMIT_CONFIG" ]; then
    echo "Creating default pre-commit configuration..."
    cat > "$PRE_COMMIT_CONFIG" << EOF
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: ["--profile", "black"]

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black
        args: ["--line-length", "88"]

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        args: ["--max-line-length", "88", "--extend-ignore", "E203"]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests]
EOF
fi

# Install the pre-commit hooks
echo "Installing pre-commit hooks..."
cd "$PROJECT_ROOT"
pre-commit install

print_success "Pre-commit hooks installed"

# Optional: Configure git
print_header "Git configuration (optional)"

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_warning "Git not found. Skipping git configuration."
else
    echo "Would you like to configure git? (y/n)"
    read -r CONFIGURE_GIT

    if [[ "$CONFIGURE_GIT" =~ ^[Yy]$ ]]; then
        # Configure user name and email if not already set
        if [ -z "$(git config --global user.name)" ]; then
            echo "Enter your name for git commits:"
            read -r GIT_NAME
            git config --global user.name "$GIT_NAME"
        else
            print_warning "Git user.name already set to: $(git config --global user.name)"
        fi

        if [ -z "$(git config --global user.email)" ]; then
            echo "Enter your email for git commits:"
            read -r GIT_EMAIL
            git config --global user.email "$GIT_EMAIL"
        else
            print_warning "Git user.email already set to: $(git config --global user.email)"
        fi

        # Enable colorful git output
        git config --global color.ui auto

        print_success "Git configured"
    else
        print_warning "Skipping git configuration"
    fi
fi

# Final instructions
print_header "Setup complete!"
echo "Your development environment is ready."
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   ${ACTIVATE_CMD}"
echo "2. Run the tests to ensure everything is working:"
echo "   pytest"
echo "3. Start developing!"
echo ""
echo "Happy coding!"