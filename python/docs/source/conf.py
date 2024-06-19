# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess
import tempfile
# sys.path.insert(0, 'C:\\Users\\jornb\\OneDrive\\Code\\getm-rewrite\\python\\pygetm')


# -- Project information -----------------------------------------------------

project = 'pygetm'
copyright = '2022, Bolding & Bruggeman ApS'
author = 'Bolding & Bruggeman ApS'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

napoleon_include_special_with_doc = True

# -- Extension configuration -------------------------------------------------
autoclass_content = 'both'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'cftime': ('https://unidata.github.io/cftime', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

if os.environ.get('READTHEDOCS', None) == 'True' and False:
    root = os.path.join(os.path.dirname(__file__), '../../..')
    outdir = tempfile.mkdtemp()
    forddir = os.path.join(outdir, 'fortran')
    os.mkdir(forddir)
    subprocess.call([sys.executable, '-m', 'ford', os.path.join(root, 'doc/getm2.md'),
        '--src_dir', os.path.join(root, 'src'), '--output_dir', forddir])
    html_extra_path = [outdir]
