#!/usr/bin/env python3
"""jig - Simple deployment tool for Together AI"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["together @ git+https://github.com/togethercomputer/together-py@next"]
# ///
import click
from together import Together

from together.lib.cli.api.beta.jig import add_commands


@click.group()
@click.pass_context
def jig(ctx: click.Context) -> None:
    """Jig - deployment tool for Together AI"""
    ctx.obj = Together()


add_commands(jig)

if __name__ == "__main__":
    jig()
