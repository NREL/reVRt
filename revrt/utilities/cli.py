"""revrt utilities command line interface (CLI)"""

from gaps.cli import CLICommandFromClass

from revrt.utilities.handlers import LayeredFile


layers_to_file = CLICommandFromClass(
    LayeredFile, method="layers_to_file", add_collect=False
)
