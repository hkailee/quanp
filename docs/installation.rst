Installation
------------

Anaconda
~~~~~~~~
If you do not have a working installation of Python 3.6, consider
installing Anaconda with Python=3.6 and create a vitualenv using conda. Then run::

    conda install seaborn scikit-learn statsmodels numba pytables
    conda install -c conda-forge python-igraph leidenalg	

The extra `python-igraph` and `leidenalg` installs two packages that are needed for popular
parts of scanpy but aren't requirements: python-igraph_ [Csardi06]_ and leiden_ [Traag18]_.

.. _python-igraph: http://igraph.org/python/
.. _leiden: https://leidenalg.readthedocs.io

Pull Quanp from `PyPI <https://pypi.org/project/quanp>`__ (consider using ``pip3`` to access Python 3)::

    pip install quanp

.. _from PyPI: https://pypi.org/project/quanp


Development Version
~~~~~~~~~~~~~~~~~~~
To work with the latest version `on GitHub`_: clone the repository and `cd` into
its root directory. To install using symbolic links (stay up to date with your
cloned version after you update with `git pull`) call::

    pip install -e .

.. _on GitHub: https://github.com/hkailee/quanp


Troubleshooting
~~~~~~~~~~~~~~~
If you get a `Permission denied` error, never use `sudo pip`. Instead, use virtual environments or::

    pip install --user quanp

**On MacOS**, if **not** using `conda`, you might need to install the C core of igraph via homebrew first

- `brew install igraph`
- If python-igraph still fails to install, see the question on `compiling igraph`_.
  Alternatively consider installing gcc via `brew install gcc --without-multilib`
  and exporting the required variables::

      export CC="/usr/local/Cellar/gcc/X.x.x/bin/gcc-X"
      export CXX="/usr/local/Cellar/gcc/X.x.x/bin/gcc-X"

  where `X` and `x` refers to the version of `gcc`;
  in my case, the path reads `/usr/local/Cellar/gcc/6.3.0_1/bin/gcc-6`.

**On Windows**, there also often problems installing compiled packages such as `igraph`,
but you can find precompiled packages on Christoph Gohlke’s `unofficial binaries`_.
Download those and install them using `pip install ./path/to/file.whl`

.. _compiling igraph: https://stackoverflow.com/q/29589696/247482
.. _unofficial binaries: https://www.lfd.uci.edu/~gohlke/pythonlibs/


Installing Anaconda
~~~~~~~~~~~~~~~~~~~~
After downloading Anaconda_, in a unix shell (Linux, Mac), run

.. code:: shell

    cd DOWNLOAD_DIR
    chmod +x Anaconda3-latest-VERSION.sh
    ./Anaconda3-latest-VERSION.sh

and accept all suggestions.
Either reopen a new terminal or `source ~/.bashrc` on Linux/ `source ~/.bash_profile` on Mac.
The whole process takes just a couple of minutes.

.. _Anaconda: https://docs.anaconda.com/anaconda/
