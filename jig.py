#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["together @ git+https://github.com/togethercomputer/together-py@next" ]
# ///
"""jig - Simple deployment tool for Together AI"""

import sys

from together.lib.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "beta", "jig", *sys.argv[1:]]
    main()
