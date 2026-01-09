"""reVRt collection CLI command"""

import glob
import shutil
import logging
import warnings
from pathlib import Path

import pandas as pd
from gaps.cli import CLICommandFromFunction
from gaps.utilities import resolve_path


from revrt.utilities import IncrementalWriter, chunked_read_gpkg
from revrt.exceptions import revrtValueError, revrtFileNotFoundError

logger = logging.getLogger(__name__)


def finalize_routes(
    collect_pattern,
    project_dir,
    out_dir,
    job_name,
    chunk_size=10_000,
    simplify_geo_tolerance=None,
    purge_chunks=False,
):
    """Merge routing output files matching a pattern into a single file

    Parameters
    ----------
    collect_pattern : str
        Unix-style ``/filepath/pattern*.gpkg`` representing the files to
        be collected into a single output file. If no output file path
        is specified (i.e. ``out_fp=None``), the output file path will
        be inferred from the  pattern itself (specifically, the wildcard
        will be removed and the result will be the output file path).
    project_dir : path-like
        Path to project directory. This path is used to resolve the
        out filepath input from the user.
    out_dir : path-like
        Directory where finalized routing file should be written.
    job_name : str
        Label used to name the generated output file.
    chunk_size : int, default=10_000
        Number of features to read into memory at a time when merging
        files. This helps limit memory usage when merging large files.
        By default, ``10_000``.
    simplify_geo_tolerance : float, optional
        Option to simplify geometries before saving to output. Note that
        this changes the path geometry and therefore create a mismatch
        between the geometry and any associated attributes (e.g.,
        length, cost, etc). This also means the paths should not be used
        for characterization. If provided, this value will be used as
        the tolerance parameter in the `geopandas.GeoSeries.simplify`
        method. Specifically, all parts of a simplified geometry will
        be no more than `tolerance` distance from the original. This
        value has the same units as the coordinate reference system of
        the GeoSeries. Only works for GeoPackage outputs
        (errors otherwise). By default, ``None``.
    out_fp : path-like, optional
        Path to output file where the merged results should be saved. If
        ``None``, the output file path will be inferred from the pattern
        itself (specifically, the wildcard will be removed and the
        result will be the output file path). By default, ``None``.
    purge_chunks : bool, default=False
        Option to delete single-node input files after the collection
        step. By default, ``False``.

    Raises
    ------
    revrtFileNotFoundError
        Raised when no files are found matching the provided pattern.
    revrtValueError
        Raised when multiple file types are found matching the pattern.
    """
    files_to_collect = _files_to_collect(collect_pattern, project_dir)

    file_suffix = _out_file_suffix(files_to_collect)
    out_fp = Path(out_dir) / f"{job_name}{file_suffix}"
    logger.info("Collecting routing outputs to: %s", out_fp)

    if simplify_geo_tolerance:
        logger.info(
            "Simplifying geometries using a tolerance of %r",
            simplify_geo_tolerance,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if file_suffix == ".gpkg":
            _collect_geo_files(
                files_to_collect,
                out_fp,
                simplify_geo_tolerance,
                chunk_size,
                purge_chunks,
            )
        else:
            _collect_csv_files(
                files_to_collect, out_fp, chunk_size, purge_chunks
            )

    return str(out_fp)


def _files_to_collect(collect_pattern, project_dir):
    """Get list of files to collect based on pattern"""
    collect_pattern = resolve_path(
        collect_pattern
        if collect_pattern.startswith("/")
        else f"./{collect_pattern}",
        project_dir,
    )

    files_to_collect = list(glob.glob(str(collect_pattern)))  # noqa
    if not files_to_collect:
        msg = f"No files found using collect pattern: {collect_pattern}"
        raise revrtFileNotFoundError(msg)

    return files_to_collect


def _out_file_suffix(files_to_collect):
    """Get the file suffix of the output file based on input files"""
    file_types = {Path(fp).suffix.lower() for fp in files_to_collect}
    if len(file_types) > 1:
        msg = (
            "Multiple file types found to collect! All files must be of "
            f"the same type. Found: {file_types}"
        )
        raise revrtValueError(msg)

    return file_types.pop()


def _collect_geo_files(
    files_to_collect, out_fp, simplify_geo_tolerance, chunk_size, purge_chunks
):
    """Collect GeoPackage files into a single output file"""
    writer = IncrementalWriter(out_fp)
    for i, data_fp in enumerate(files_to_collect, start=1):
        logger.info("Loading %s (%i/%i)", data_fp, i, len(files_to_collect))
        for df in chunked_read_gpkg(data_fp, chunk_size):
            if simplify_geo_tolerance:
                df.geometry = df.geometry.simplify(simplify_geo_tolerance)

            writer.save(df)

        _handle_chunk_file(Path(out_fp).parent, data_fp, purge_chunks)


def _collect_csv_files(files_to_collect, out_fp, chunk_size, purge_chunks):
    """Collect CSV files into a single output file"""
    writer = IncrementalWriter(out_fp)
    for i, data_fp in enumerate(files_to_collect, start=1):
        logger.info("Loading %s (%i/%i)", data_fp, i, len(files_to_collect))
        logger.debug(
            "\t- Processing CSV in chunks of %d",
            chunk_size,
        )
        for chunk_idx, df in enumerate(
            pd.read_csv(data_fp, chunksize=chunk_size)  # cspell:disable-line
        ):
            logger.debug("\t\t- Processing CSV chunk %d", chunk_idx)
            if len(df) == 0:
                continue
            writer.save(df)

        _handle_chunk_file(Path(out_fp).parent, data_fp, purge_chunks)


def _handle_chunk_file(out_dir, chunk_fp, purge_chunks):
    """Handle chunk file after collection step"""
    chunk_fp = Path(chunk_fp)
    if purge_chunks:
        logger.info("Purging chunk file: %s", chunk_fp)
        chunk_fp.unlink()
    else:
        logger.debug("Retaining chunk file: %s", chunk_fp)
        new_dir = out_dir / "chunk_files"
        new_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(chunk_fp, new_dir / chunk_fp.name)


finalize_routes_command = CLICommandFromFunction(
    finalize_routes,
    name="finalize-routes",
    add_collect=False,
)
