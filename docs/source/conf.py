# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'ElfinAndError'
copyright = '2024, firstelfin'
author = 'firstelfin'

release = '0.1.1'
version = '0.1.1'

# -- General configuration

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid'
]

html_js_files = [
    'https://unpkg.com/mermaid/dist/mermaid.min.js'
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

language = 'zh_CN'
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
