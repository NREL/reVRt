"""revrt command line interface (CLI)"""

import logging

from gaps.cli import make_cli

from revrt import __version__
from revrt.spatial_characterization.cli import route_characterizations_command


logger = logging.getLogger(__name__)


commands = [route_characterizations_command]
main = make_cli(commands, info={"name": "reVRt", "version": __version__})

# export GAPs commands to namespace for documentation
route_characterization = main.commands["route-characterization"]


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        logger.exception("Error running reVRt CLI")
        raise
