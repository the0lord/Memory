"""Launcher for the solmem MCP server that works from any working directory.

`claude mcp add` may start the server with an arbitrary cwd, so we put the repo
on sys.path here rather than relying on `py -m solmem.server`.

    claude mcp add solmem -- py D:\\Fun\\Memory\\solmem_server.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from solmem.server import main  # noqa: E402

if __name__ == "__main__":
    main()
