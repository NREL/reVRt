"""TreV command line interface (CLI)"""

import logging

from gaps.cli import make_cli

from trev._version import __version__
from trev.spatial_characterization.cli import stats_command


logger = logging.getLogger(__name__)


commands = [stats_command]
main = make_cli(commands, info={"name": "trev", "version": __version__})

# export GAPs commands to namespace for documentation
stats = main.commands["stats"]


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        logger.exception("Error running TreV CLI")
        raise
