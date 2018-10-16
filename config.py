#!/usr/bin/env python2
"""Configuration values for the project."""
import os.path as path
import click

############################ EDIT HERE ########################################
UP_FP = path.expanduser("external/up")
SMPL_FP = path.expanduser("external/smpl")
EIGEN_FP = "/usr/include/eigen3/"

###############################################################################
# Infrastructure. Don't edit.                                                 #
###############################################################################

@click.command()
@click.argument('key', type=click.STRING)
def cli(key):
    """Print a config value to STDOUT."""
    if key in globals().keys():
        print globals()[key]
    else:
        raise Exception("Requested configuration value not available! "
                        "Available keys: " +
                        str([kval for kval in globals().keys()
                             if kval.isupper()]) +
                        ".")


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter

