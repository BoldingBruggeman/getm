# pygetm

This is a rewrite of the [General Estuarine Transport Model (GETM)](https://getm.eu).
It is mostly written in Python; only performance-critical sections of the code are
implemented in Fortran.

## Installing

First, ensure you have [Anaconda](https://docs.anaconda.com/free/anaconda/):
   - Linux/Mac: execute `conda --version` in a terminal
   - Windows: look for "Anaconda prompt" in the start menu

On some systems (notably, HPC clusters), you may need to load an anaconda module first:
try `module load anaconda` or `module load anaconda3`

If you do *not* have Anaconda, install [Miniconda](https://docs.anaconda.com/miniconda/).

From here on, we will be working in a terminal window. On Windows, open a terminal by choosing "Anaconda prompt" in the start menu.

### Install a prebuilt version from conda-forge

To install or update pygetm:

```
conda install -c conda-forge pygetm
```

### Manual build and install

If you need a customized version of pygetm, for instance, built with specific compiler
options, or with specific biogeochemical models that are not part of the standard
[FABM](https://fabm.net) distribution, you can manually obtain the pygetm source code,
build it, and then install it.

To obtain the repository with setups and scripts, first set up and activate a conda
environment with all necessary build tools:

```
git clone --recursive https://github.com/BoldingBruggeman/getm-rewrite.git
cd getm-rewrite
conda env create -f environment.yml
conda activate pygetm
```

If you are installing on an system that already has a Fortran
compiler and MPI libraries that you would like to use, replace `environment.yml` with
`environment-min.yml` in the above.

The above requires that you already have [Git](https://git-scm.com/) installed.
If you do not, you can install this with `conda install -c conda-forge git`.

Finally, to build on Linux/Mac, execute

```
source ./install
```

If you are using a different shell than bash, you may need to replace `source` by `bash`.

On Windows, you build pygetm with

```
install.bat
```

You can customize the build step as follows:
* To set the Fortran compiler, set environment variable `FC` to your desired Fortran
  compiler executable. For instance, in a bash shell: `export FC=ifort`. This can be done
  in your terminal before running the install script, or by adding it to the install
  script itself.
* To set compilation flags, set environment variable `FFLAGS` to your desired flags.
  For instance, in a bash shell: `export FFLAGS=-fcheck=all`. This can be done in your
  terminal before running the install script, or by adding it to the install script
  itself.
* To set cmake options used to compile FABM, such as `-DFABM_EXTRA_INSTITUTES` or
  `-DFABM_<INSTITUTE>_BASE`, add `cmake_opts=<CMAKE_OPTIONS>` to [`python/setup.cfg`](https://github.com/BoldingBruggeman/getm-rewrite/blob/devel/python/setup.cfg)

#### Staying up to date

To update this repository including its submodules (GOTM, FABM, etc.), make sure you are
in the getm-rewrite directory and execute:

```
git pull
git submodule update --init --recursive
conda env update -f <ENVIRONMENT_YML>
conda activate pygetm
```

In the above, replace `<ENVIRONMENT_YML>` with the name of the environment file you used
previously: `environment.yml` for stand-alone conda environments, or `environment-min.yml`
for a setup that uses the local MPI implementation and Fortran compiler.

Finally, rebuild by executing `source ./install` on Linux/Mac, or `install.bat` on Windows.

## Using pygetm

You should always activate the correct Python environment before you use the model with
`conda activate pygetm`. This needs to be done any time you start a new shell.

### Jupyter Notebooks

The best place to start is the [`python/examples`](https://github.com/BoldingBruggeman/getm-rewrite/tree/devel/python/examples)
directory with Jupyter Notebooks that demonstrate the functionality of the model:

```
cd python/examples
python -m jupyterlab
```

### Simulations

Some of the original GETM test cases have been ported to pygetm:

* [north_sea](https://github.com/BoldingBruggeman/getm-rewrite/blob/devel/python/examples/north_sea_legacy.py)
  - including [an extended version](https://github.com/BoldingBruggeman/getm-rewrite/blob/devel/python/examples/north_sea.py)
  that shows new pygetm features such as command-line configurability.
* [box_spherical](https://github.com/BoldingBruggeman/getm-rewrite/blob/devel/python/examples/box_spherical.py)
* [seamount](https://github.com/BoldingBruggeman/getm-rewrite/blob/devel/python/examples/seamount.py)

To run a simulation:

```
python <RUNSCRIPT.py> [OPTIONS]
```

To run in parallel:

```
mpiexec -n <NCPUS> python <RUNSCRIPT.py> [OPTIONS]
```

## Contributing

How to contribute to the development:

  1. [Make a fork](https://github.com/BoldingBruggeman/getm-rewrite/fork) of the
     repository under your private GitHub account(\*)
  2. Commit your changes to your forked repository
  3. Make a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)

Note that all communication in relation to development of GETM is done via
GitHub using [issues](https://github.com/BoldingBruggeman/tickets/issues).


(\*) If you use a service other than GitHub for your daily work - please have a
look [here](https://stackoverflow.com/questions/37672694/can-i-submit-a-pull-request-from-gitlab-com-to-github)

https://yarchive.net/comp/linux/collective_work_copyright.html
