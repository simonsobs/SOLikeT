# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Import SOLikeT (for autodoc)
import sys
sys.path.insert(0, "..")

# Create some mock imports
import mock
MOCK_MODULES = ["cosmopower", "tensorflow", "pyccl", "camb"]
for module in MOCK_MODULES:
    sys.modules[module] = mock.Mock()

import soliket

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SOLikeT'
copyright = '2023, The SO Collaboration'
author = 'The SO Collaboration'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc", # Generate doc pages from source docstrings
    "sphinx.ext.viewcode", # Generate links to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
