"""Sphinx configuration."""

from importlib.metadata import version as get_version

project = "flu-peak-evt"
author = "Cameron Maloney"

try:
    release = get_version("flu-peak-evt")
except Exception:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
