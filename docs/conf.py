"""Sphinx configuration for Macrodata Refinement documentation."""

import os
import sys
from datetime import datetime

# Add the project root and src directories to the path
sys.path.insert(0, os.path.abspath('..')) 
sys.path.insert(0, os.path.abspath('../src'))

# Mock imports for libraries that may not be available during docs build
import importlib.util

# Check if we need to use mock imports (e.g., on ReadTheDocs)
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd or not importlib.util.find_spec('numpy'):
    from unittest.mock import MagicMock

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return MagicMock()

    # Mock these modules for autodoc
    MOCK_MODULES = ['numpy', 'pandas', 'matplotlib', 'matplotlib.pyplot', 'scipy', 'scipy.stats']
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# Project information
project = 'Macrodata Refinement'
copyright = f'{datetime.now().year}, Mudit Bhargava'
author = 'Mudit Bhargava'

# The full version, including alpha/beta/rc tags
release = '0.1.0'
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns to exclude from source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'images/mdr_logo.svg'
html_favicon = 'images/mdr_logo.svg'
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_preprocess_types = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# AutoAPI settings
autoapi_type = 'python'
autoapi_dirs = ['../src/mdr']
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary']
autoapi_add_toctree_entry = False
