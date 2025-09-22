"""revrt command line interface (CLI)"""

import logging

from gaps.cli import make_cli

from revrt import __version__
from revrt.spatial_characterization.cli import route_characterizations_command
from revrt.costs.cli import build_routing_layers_command
from revrt.utilities.cli import (
    layers_to_file_command,
    layers_from_file_command,
)


logger = logging.getLogger(__name__)


commands = [
    layers_to_file_command,
    layers_from_file_command,
    build_routing_layers_command,
    route_characterizations_command,
]
main = make_cli(commands, info={"name": "reVRt", "version": __version__})


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        logger.exception("Error running reVRt CLI")
        raise
