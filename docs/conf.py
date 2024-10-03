import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../fusemap'))

project = 'AutoSort'
copyright = '2024, Yichun He'
author = 'Yichun He'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",]



# autodoc configuration
autodoc_typehints = "description"
autodoc_mock_imports = ["anndata",
                        "hnswlib",
                        "captum",
                        "circlify",
                        "matplotlib",
                        "networkx",
                        "numba",
                        "numcodecs",
                        "numpy",
                        "obonet",
                        "pandas",
                        "pegasusio",
                        "pytorch_lightning",
                        "scanpy",
                        "scipy",
                        "seaborn",
                        "tiledb",
                        "tqdm",
                        "torch",
                        "zarr"]

# todo configuration
todo_include_todos = True


# nbsphinx configuration
nbsphinx_thumbnails = {
    'notebooks/spatial_integrate_tech': '_static/test.png',
    'notebooks/spatial_integrate_species': '_static/test.png',
    'notebooks/spatial_impute': '_static/test.png',
    'notebooks/spatial_map_mousebrain': '_static/test.png',
    'notebooks/spatial_map_mousehuman': '_static/test.png'
}


templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True


# html_theme = 'sphinx_rtd_theme'

html_theme = "sphinx_book_theme"
# autodoc_class_signature = "separated"

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# Custom CSS
html_css_files = ['custom.css']


# Output file base name for HTML help builder.
htmlhelp_basename = "fusemap-doc"