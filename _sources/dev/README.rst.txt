.. _contributing:

Contributing to reVRt
=====================

This document discusses working on the reVRt code base and documentation.


Overview
--------

* All code changes should be submitted via a pull request (PR) and
  reviewed/approved by a core team member before merging  (see `Contributing Process`_ below for more details).
* Before setting up your environment, see the `Development environment guidelines`_.
* All code should adhere to `PEP8 <https://peps.python.org/pep-0008/>`_ (if you haven't
  read over it in a while, we recommend you skim it again for a refresher).
* All code should adhere to the `Stylistic guidelines`_.
* All code should have tests (see `Test coverage`_ below for more details).
* All code should be documented (see `Documentation`_ below for more details).


Contributing Process
--------------------

If you need a refresher on contributing code via GitHub using a pull request, check out the
`official GitHub documentation <https://docs.github.com/en/pull-requests>`_. There, you can
learn how to `create a branch <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository>`_,
`open a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_,
and `request a review <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review>`_.

If you are not sure where or how to start contributing to reVRt, consider reaching out to the
current developer(s): **Paul Pinchuk** ["ppinchuk@nrel.gov"] or **Guilherme Pimenta Castelao** ["gpimenta@nrel.gov"].

When you are ready to contribute to reVRt, clone a copy of the reVRt repository from GitHub, check out your
own branch, and make the appropriate changes locally. Ensure that your new code adheres to all of the guidelines
below. When you are happy with your code, commit it locally. Ensure that you commit all tests that go along
with your code change.

Once all your code changes are committed locally and your code passes all tests, push it to the GitHub repository
and open a PR. Give your PR a short title and briefly describe your changes. Request that one of the core reVRt
developers reviews your PR. We will likely ask you to make some modifications before you can merge. This is typical
and ensures that the quality of the overall codebase remains somewhat consistent.

Once your PR is approved, you may merge it into the main branch. If your code introduces significant new functionality
or fixes a critical bug, it may warrant a release. Please coordinate with a core reVRt developer to create a new release,
which will automatically generate new wheels for installing reVRt.


.. index-dev-link-end:

.. _dev-env-guidelines:
.. index-dev-link-start:


Development environment guidelines
----------------------------------

We use `pixi <https://pixi.sh/latest/>`_ to manage environments across developers.
This tool allows developers to install libraries and applications in a reproducible
way across multiple platforms. This means bugs are easier to reproduce, and it's easier
to move your development environment to a new piece of hardware.

We keep a version-controlled ``pixi.lock`` in the repository to allow locking with
the full requirements tree so we can reproduce behaviors and bugs and easily compare
results.

You can use the ``pdev`` feature in ``pixi`` to get all necessary python development tools:

.. code-block:: shell

    pixi shell -e pdev

To work on the Rust-based CLI, you can use the ``rdev`` feature instead:

.. code-block:: shell

    pixi shell -e rdev

You are welcome to use a different environment manager (e.g. ``conda``, ``mamba``, etc),
but we make no promises to provide support on environment-related issues/bugs in this case.


Stylistic guidelines
--------------------

We define a set of stylistic standards for reVRt code development. The intent
is to maintain coherence when multiple developers contribute code to the repository.

Ruff
^^^^

Maintaining consistent code quality is crucial for reVRt. To ensure uniformity and
adherence to coding standards, we employ the use `Ruff <https://docs.astral.sh/ruff/>`_.
Ruff is an "opinionated" formatter and linter designed to enhance code readability,
maintainability, and consistency that is extremely fast.

You can use the `Ruff VSCode extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_
if you are developing using VSCode. If you do so, make sure to set the path to the ``ruff``
executable in your VSCode settings to point to the ``ruff`` installed in the ``dev`` environment.
You can do this by adding the following to your ``settings.json`` file:

.. code-block:: json

    "ruff.path": [
        "${workspaceFolder}/.pixi/envs/dev/bin/ruff"
    ]

Alternatively, you can set a pre-commit hook to run Ruff. This would perform automatic code formatting
before any code is committed to the repository.

Both of these tools ensure that all code contributions meet the established quality standards,
minimizing the chances of introducing formatting inconsistencies or potential issues.


Imports
^^^^^^^

Use the following import conventions::

   import gaps
   import rasterio
   import rioxarray
   import numpy as np
   import pandas as pd
   import xarray as xr
   import dask.array as da


Test coverage
-------------

Pull requests (PRs) that modify code should either have new tests, or modify existing
tests to fail before the PR and pass afterwards.

You can run python reVRt tests locally using ``pixi``:

.. code-block:: shell

    pixi r tests

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%, though this alone does not ensure that
your code is bug-free. Still, this is a good place to start, and you view the test
coverage at ``build/coverage`` by running:

.. code-block:: shell

    firefox build/coverage/index.html


Documentation
-------------

We strongly believe that documentation is a core part of code development, as it helps
both users of your function as well as other developers (including your future self).
As such, please adhere to these guidelines:

1) Document all public functions and classes
    Public functions and classes are defined as not having any leading underscores (``_``).
    These functions are detected by Sphinx and therefore should have docstrings formatted according to the
    `NumPy documentation style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
2) Do not include a period (``.``) on the first line (short summary) of a docstring.
    This is a stylistic decision particular to the reVRt codebase.
3) Do not include a short summary (first line docstring) for ``__init__`` methods.
    Instead, document any object summaries using the class docstring. You can and should still document
    initialization parameters in the ``__init__`` docstring. See any the docstring of the main reVRt objects
    (e.g. :class:`~revrt.spatial_characterization.zonal.ZonalStats`) for an example.
4) Protected/private functions should contain minimal documentation.
    Public functions and classes are defined as having one or more leading underscores (``_``).
    These functions are **not** detected by Sphinx and therefor should contain minimal documentation
    (typically a docstring with just a single sentence). Do not include *any* sections from the
    NumPy documentation style. With minimal exceptions,
    we treat protected and private functions as implementation details. As such, if you did not
    write the function, you should probably not be modifying/calling/touching it in any way.
    Such code is subject to change at any time, so you should never rely on private/protected
    functionality unless you know what you are doing (in which case you should be relying on the
    function's code, not docstring).
5) Link any functions and/or classes that you reference in your docstring.
    Sphinx allows interlinks between different sets of documentation, which can be a really convenient
    way for new users to learn more about the external libraries they are expected to use. For more
    information on how to set up links in your documentation, please see
    `this short blog post <https://kevin.burke.dev/kevin/sphinx-interlinks/>`_. In particular,
    we use the ``:func:`` directive for standalone functions, ``:meth:`` for class methods,
    ``:class:`` for references to classes, and ``:obj:`` for all other links. Please use this
    list of available reVRt intersphinx mappings:

        * reVRt: ``revrt``
            For example, use ``:func:`~revrt.utilities.base.buffer_routes```,
            which renders as :func:`~revrt.utilities.base.buffer_routes`
        * Numpy: ``numpy``
            For example, use ``:obj:`~numpy.array```, which renders as :obj:`~numpy.array`
        * dask: ``dask``
            For example, use ``:func:`dask.compute```, which renders as :func:`dask.compute`
        * dask.distributed: ``distributed``
            For example, use ``:class:`~distributed.Client```, which renders as :class:`~distributed.Client`
        * gaps: ``gaps``
            For example, use ``:class:`~gaps.cli.command.CLICommandFromFunction```, which renders as :class:`~gaps.cli.command.CLICommandFromFunction`
        * GeoPandas: ``geopandas``
            For example, use ``:class:`geopandas.GeoDataFrame```, which renders as :class:`geopandas.GeoDataFrame`
        * ODC-Geo: ``odc.geo``
            For example, use ``:class:`~odc.geo.geobox.GeoBox```, which renders as :class:`~odc.geo.geobox.GeoBox`
        * Shapely: ``shapely``
            For example, use ``:class:`~shapely.Point```, which renders as :class:`~shapely.Point`
        * Rasterio: ``rasterio``
            For example, use ``:class:`~rasterio.io.DatasetReader```, which renders as :class:`~rasterio.io.DatasetReader`
        * RioXarray: ``rioxarray``
            For example, use ``:class:`~rioxarray.raster_array.RasterArray```, which renders as :class:`~rioxarray.raster_array.RasterArray`
        * Xarray: ``xarray``
            For example, use ``:func:`~xarray.open_dataset```, which renders as :func:`~xarray.open_dataset`
        * MatplotLib: ``matplotlib``
            For example, use ``:func:`~matplotlib.pyplot.plot```, which renders as :func:`~matplotlib.pyplot.plot`
        * Plotly: ``plotly``
            For example, use ``:class:`plotly.graph_objects.Figure```, which renders as :class:`plotly.graph_objects.Figure`


import gaps
   import rasterio
   import rioxarray
   import numpy as np
   import pandas as pd
   import xarray as xr
   import dask.array as da


To check your docstring additions/updates, you can build a local version of the HTML documentation:

.. code-block:: shell

    pixi r make-html

After running this command, simply open ``docs/_build/html/index.html`` using your favorite browser, e.g.:

.. code-block:: shell

    firefox docs/_build/html/index.html


Miscellaneous
-------------

A collection of other miscellaneous guidelines.


GitHub Actions Cache and Updating ``pyproject.toml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because we statically link the requirements, the Rust compilation process can
extend to 30-60 minutes. That is mostly due to the ``duckdb`` and ``tokio``
crates. To optimize this process, we use a GitHub Actions cache.

When using the GitHub cache system, we have to be mindful of the 10 GB total
storage limit. If we place too many items in the cache, it will rotate too
frequently and defeat the entire purpose of the cache. For this reason,
**we only cache environments that are run in actions on the main branch**!

With this system, any PR can then pull from the cache built on the main branch
and set up their environments that way.

What this means for you
"""""""""""""""""""""""
When you open a PR, your environment will be built from a cache from the main branch.
If you have no dependency updates, you are good to go!

However, if you do have dependency updates, your environment will need to be updated.
If you are working with Rust, you will download and compile the extra crate(s) in your
branch. If the crate is small, this may not be a big deal, but keep in mind that this
will happen for every new commit you push to your open PR.

If you updated something in the ``pixi`` environment, the whole environment will be re-built.

Therefore, in both of the latter cases, a good practice is to put your dependency updates
in a separate branch and dedicated PR that you merge to main. Then, your feature PR
can make full use of the cache that is built on the main branch without having to re-build
or re-compile anything for the environment.


Error Handling
^^^^^^^^^^^^^^

Do not throw default warning/errors. Always use some subclass of
``revrt.warn.revrtWarning`` or ``revrt.exceptions.revrtError``, like so::

    from warnings import warn

    from revrt.warn import revrtWarning
    from revrt.exceptions import revrtValueError

    ...

    def my_func():
        if not_good_enough_data:
            warn("Watch out for your data!", revrtWarning)

        if unacceptable_value:
            raise revrtValueError("This value is unacceptable")


This allows maximum flexibility for downstream users of the library.
In particular, they can choose whether to handle just errors that originate
from reVRt (i.e. by catching ``revrtValueError``) or to handle the generic
version of the error (i.e. by catching ``ValueError``).

In addition, all reVRt warnings and errors have a logging call built in.
This means you no longer need to do something like this::

    # BAD - do not do this
    if unacceptable_value:
        msg = "This value is unacceptable"
        logger.exception(msg)
        raise ValueError(msg)

Simply raising ``revrtValueError`` with the appropriate message performs the
logging call shown above, internally, every time.


Test File Structure
^^^^^^^^^^^^^^^^^^^

All test files (e.g. ``test_scenario.py``) should start/end with the following block of code:

.. code-block:: python

    from pathlib import Path
    import pytest

    ...

    if __name__ == "__main__":
        pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])


This allows the (single) file to be executed, running only the tests contained
within. This is extremely useful when updating/modifying/adding tests in the file.


Versioning
^^^^^^^^^^
This repository uses `Semantic Versioning <https://semver.org/>`_ for its three
main components:

* ``rX.X.X``: Rust core library version
* ``cX.X.X``: Rust command-line interface (CLI) version
* ``vX.X.X``: Python package version

These version numbers should match the major changes in each software release.
For example, if you make a change to the Rust core library that does not affect the CLI,
you would increment the ``rX.X.X`` version number, but not the ``cX.X.X`` or
``vX.X.X`` version numbers.


Rust Benchmarking
^^^^^^^^^^^^^^^^^
To ensure the routing algorithm remains performant, we have set up
`Rust benchmarks <https://nrel.github.io/reVRt/dev/bench/index.html>`_
that are automatically run on GitHub Actions on the main branch. If
you are actively developing the Rust-based routing algorithm, you
should monitor the benchmarks to ensure that your changes do not
introduce performance regressions.
