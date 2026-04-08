#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["together @ git+https://github.com/togethercomputer/together-py@v2.2.1-alpha0" ]
# ///
"""jig - Simple deployment tool for Together AI"""

from together.lib.cli.api.beta.jig import jig

if __name__ == "__main__":
    jig()
