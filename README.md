# pygetm

This is a rewrite of the [General Estuarine Transport Model (GETM)](https://getm.eu).
It is mostly written in Python; only performance-critical sections of the code are
implemented in Fortran.

## Installing

You will need [Anaconda](https://docs.anaconda.com/free/anaconda/). On many systems
that is already installed: try running `conda --version`. If that fails, you may need
to load an anaconda module first: try `module load anaconda` or `module load anaconda3`.
If that still does not give you a working `conda` command, you may want to install
[Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).

Before using conda for the very first time, you will need to initialize its environment:

```
conda init bash
```

If you are using a different shell than bash, replace `bash` with the name of your shell
(see `conda init -h` for supported ones), or use `conda init --all`.

This needs to be done just once, as it modifies your `.bashrc` that is sourced every time
you login. After this, restart your shell by logging out and back in.

### Installation with conda (currently Linux/Windows only)

To install or update pygetm:

```
conda install pygetm -c bolding-bruggeman -c conda-forge
```

### Manual build and install

If you need a customized version of pygetm, for instance, built with specific compiler
options, or with specific biogeochemical models that are not part of the standard
[FABM](https://fabm.net) distribution, you can manually obtain the pygetm source code,
build it, and then install it.

#### Linux/Mac

To obtain the repository with setups and scripts, set up your conda environment, and
build and install pygetm:

```
git clone --recursive https://github.com/BoldingBruggeman/getm-rewrite.git
cd getm-rewrite
conda env create -f environment.yml
conda activate pygetm
source ./install
```

If you are using a different shell than bash, you may need to replace `source` in the
last line  by `bash`. If you are installing on an HPC system that already has a Fortran
compiler and MPI libraries that you would like to use, replace `environment.yml` with
`environment-min.yml` in the above.

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

#### Windows

As on other platforms, you need [Anaconda](https://docs.anaconda.com/free/anaconda/)
or [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/). In addition,
you need to ensure that software to obtain and build Fortran code is available.
Therefore, install:

* [Git for Windows](https://git-scm.com/download/win)
* [Visual Studio Community 2019](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads)
* [Intel Fortran Compiler](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#fortran)
* [Microsoft MPI](https://github.com/microsoft/Microsoft-MPI/releases) - you need both the runtime library and the Software Development Kit

Now obtain the repository with setups and scripts, set up your conda environment,
and build and install pygetm:

```
git clone --recursive https://github.com/BoldingBruggeman/getm-rewrite.git
cd getm-rewrite
conda env create -f environment-min.yml
conda activate pygetm
install.bat
```

#### Staying up to date

To update this repository including its submodules (GOTM, FABM, etc.), make sure you are
in the getm-rewrite directory and execute:

```
git pull
git submodule update --init --recursive
conda env update -f <ENVIRONMENT_YML>
conda activate pygetm
source ./install
```

In the above, replace `<ENVIRONMENT_YML>` with the name of the environment file you used
previously: `environment.yml` for stand-alone conda environments, or `environment-min.yml`
for a setup that uses the local MPI implementation and Fortran compiler.

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
